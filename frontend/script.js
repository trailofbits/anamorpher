class AnamorpherApp {
    constructor() {
        this.baseURL = 'http://localhost:5000';
        this.currentImage = null;
        this.currentImageDimensions = null;
        this.downsamplers = {};
        this.currentTab = 'downsampling';
        this.decoyImages = [];
        this.demoImages = [];
        this.currentTextPreview = null;
        
        this.initElements();
        this.initEventListeners();
        this.loadConfiguration();
    }
    
    initElements() {
        this.elements = {
            // Downsampling elements
            imageInput: document.getElementById('imageInput'),
            uploadStatus: document.getElementById('uploadStatus'),
            downsamplerSelect: document.getElementById('downsamplerSelect'),
            methodSelect: document.getElementById('methodSelect'),
            processBtn: document.getElementById('processBtn'),
            results: document.getElementById('results'),
            loading: document.getElementById('loading'),
            error: document.getElementById('error'),
            errorMessage: document.getElementById('errorMessage'),
            metadata: document.getElementById('metadata'),
            originalImage: document.getElementById('originalImage'),
            downsampledImage: document.getElementById('downsampledImage'),
            sliderHandle: document.getElementById('sliderHandle'),
            comparisonSlider: document.getElementById('comparisonSlider'),

            // Adversarial elements
            targetText: document.getElementById('targetText'),
            generateTextBtn: document.getElementById('generateTextBtn'),
            textPreview: document.getElementById('textPreview'),
            textPreviewImg: document.getElementById('textPreviewImg'),
            textWarning: document.getElementById('textWarning'),
            fontSizeInput: document.getElementById('fontSizeInput'),
            textAlignInput: document.getElementById('textAlignInput'),
            decoySelect: document.getElementById('decoySelect'),
            methodSelect2: document.getElementById('methodSelect2'),
            lamInput: document.getElementById('lamInput'),
            epsInput: document.getElementById('epsInput'),
            gammaInput: document.getElementById('gammaInput'),
            darkFracInput: document.getElementById('darkFracInput'),
            darkFracInputBilinear: document.getElementById('darkFracInputBilinear'),
            offsetInput: document.getElementById('offsetInput'),
            generateAdvBtn: document.getElementById('generateAdvBtn'),
            adversarialResults: document.getElementById('adversarialResults'),
            advMetadata: document.getElementById('advMetadata'),
            originalDecoy: document.getElementById('originalDecoy'),
            targetImage: document.getElementById('targetImage'),
            adversarialImage: document.getElementById('adversarialImage'),
            bicubicParams: document.getElementById('bicubicParams'),
            bilinearParams: document.getElementById('bilinearParams'),
            nearestParams: document.getElementById('nearestParams'),

            // Tab elements
            navTabs: document.querySelectorAll('.nav-tab'),
            tabContents: document.querySelectorAll('.tab-content'),

            // Demo elements
            demoButton: document.getElementById('demoButton'),
            demoDropdown: document.getElementById('demoDropdown'),
            demoPreview: document.getElementById('demoPreview'),
            demoPreviewImg: document.getElementById('demoPreviewImg')
        };
    }
    
    initEventListeners() {
        // Downsampling tab listeners
        this.elements.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.elements.downsamplerSelect.addEventListener('change', (e) => this.handleDownsamplerChange(e));
        this.elements.processBtn.addEventListener('click', () => this.processImage());
        this.elements.comparisonSlider.addEventListener('input', (e) => this.updateComparison(e));
        
        // Adversarial tab listeners
        this.elements.generateTextBtn.addEventListener('click', () => this.generateTextPreview());
        this.elements.generateAdvBtn.addEventListener('click', () => this.generateAdversarial());
        this.elements.methodSelect2.addEventListener('change', (e) => this.handleMethodChange(e));
        this.elements.targetText.addEventListener('input', () => this.updateGenerateAdvButton());
        this.elements.decoySelect.addEventListener('change', () => this.updateGenerateAdvButton());
        this.elements.fontSizeInput.addEventListener('change', () => this.hideTextPreview());
        this.elements.textAlignInput.addEventListener('change', () => this.hideTextPreview());

        // Tab navigation
        this.elements.navTabs.forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Demo image selection
        this.elements.demoButton.addEventListener('click', (e) => this.toggleDemoDropdown(e));
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.demo-button-container')) {
                this.closeDemoDropdown();
            }
        });
        
        // Add mouse event listeners for draggable slider handle
        this.initSliderDrag();
    }
    
    initSliderDrag() {
        let isDragging = false;
        
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            
            const imageContainer = document.querySelector('.image-container');
            if (!imageContainer) return;
            
            const rect = imageContainer.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
            
            // Update the slider value and trigger comparison update
            this.elements.comparisonSlider.value = percentage;
            this.updateComparison({ target: { value: percentage } });
        };
        
        const handleMouseUp = () => {
            isDragging = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
        };
        
        // Add mousedown event to slider handle
        this.elements.sliderHandle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            document.body.style.cursor = 'ew-resize';
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        });
        
        // Image container click will be added in showResults when container exists
    }
    
    async loadConfiguration() {
        try {
            // Load downsamplers
            const downsamplersResponse = await fetch(`${this.baseURL}/api/downsamplers`);
            this.downsamplers = await downsamplersResponse.json();
            this.populateDownsamplers();
            

            // Load decoy images
            await this.loadDecoyImages();

            // Load demo images
            await this.loadDemoImages();
            
        } catch (error) {
            this.showError('Failed to load configuration. Make sure the backend server is running.');
        }
    }
    
    populateDownsamplers() {
        this.elements.downsamplerSelect.innerHTML = '<option value="">Select downsampler...</option>';
        
        for (const [key, downsampler] of Object.entries(this.downsamplers)) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = downsampler.name;
            this.elements.downsamplerSelect.appendChild(option);
        }
    }
    
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) {
            this.currentImage = null;
            this.currentImageDimensions = null;
            this.elements.uploadStatus.textContent = 'No image selected';
            this.updateProcessButton();
            return;
        }
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Validate dimensions - must be square and divisible by 4
                if (img.width !== img.height) {
                    this.showError(`Image must be square. Got ${img.width}x${img.height}.`);
                    this.currentImage = null;
                    this.currentImageDimensions = null;
                    this.elements.uploadStatus.textContent = 'Invalid image dimensions';
                    return;
                }
                if (img.width % 4 !== 0) {
                    this.showError(`Image dimensions must be divisible by 4. Got ${img.width}x${img.height}.`);
                    this.currentImage = null;
                    this.currentImageDimensions = null;
                    this.elements.uploadStatus.textContent = 'Invalid image dimensions';
                    return;
                }
                
                this.currentImage = e.target.result;
                this.currentImageDimensions = { width: img.width, height: img.height };
                const targetResolution = img.width / 4;
                this.elements.uploadStatus.textContent = `${file.name} (${img.width}x${img.height} → ${targetResolution}x${targetResolution})`;
                
                // Hide demo preview when file is uploaded
                this.elements.demoPreview.style.display = 'none';
                
                this.hideError();
                this.updateProcessButton();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    handleDownsamplerChange(event) {
        const selectedDownsampler = event.target.value;
        
        // Clear and disable method select
        this.elements.methodSelect.innerHTML = '<option value="">Select method...</option>';
        this.elements.methodSelect.disabled = !selectedDownsampler;
        
        if (selectedDownsampler && this.downsamplers[selectedDownsampler]) {
            // Populate methods for selected downsampler
            const methods = this.downsamplers[selectedDownsampler].methods;
            methods.forEach(method => {
                const option = document.createElement('option');
                option.value = method;
                option.textContent = method.charAt(0).toUpperCase() + method.slice(1);
                this.elements.methodSelect.appendChild(option);
            });
            this.elements.methodSelect.disabled = false;
        }
        
        this.updateProcessButton();
    }
    
    updateProcessButton() {
        const hasImage = this.currentImage !== null;
        const hasDownsampler = this.elements.downsamplerSelect.value !== '';
        const hasMethod = this.elements.methodSelect.value !== '';
        
        this.elements.processBtn.disabled = !(hasImage && hasDownsampler && hasMethod);
    }

    // Toggle demo dropdown
    toggleDemoDropdown(event) {
        event.stopPropagation();
        const isShown = this.elements.demoDropdown.classList.contains('show');
        if (isShown) {
            this.closeDemoDropdown();
        } else {
            this.showDemoDropdown();
        }
    }

    showDemoDropdown() {
        this.elements.demoDropdown.classList.add('show');
    }

    closeDemoDropdown() {
        this.elements.demoDropdown.classList.remove('show');
    }

    // Load available demo images
    async loadDemoImages() {
        try {
            const response = await fetch(`${this.baseURL}/api/demo-images`);
            this.demoImages = await response.json();
            this.populateDemoImages();
        } catch (error) {
            console.error('Failed to load demo images:', error);
        }
    }

    populateDemoImages() {
        this.elements.demoDropdown.innerHTML = '';
        
        this.demoImages.forEach(demo => {
            const option = document.createElement('div');
            option.className = 'demo-option';
            option.textContent = `${demo.filename} (${demo.width}x${demo.height})`;
            option.dataset.filename = demo.filename;
            option.addEventListener('click', (e) => this.selectDemoImage(e));
            this.elements.demoDropdown.appendChild(option);
        });
    }

    selectDemoImage(event) {
        const filename = event.target.dataset.filename;
        this.closeDemoDropdown();
        this.loadDemoImage(filename);
    }

    // Load selected demo image
    async loadDemoImage(filename) {
        if (!filename) {
            this.currentImage = null;
            this.currentImageDimensions = null;
            this.elements.demoPreview.style.display = 'none';
            this.elements.uploadStatus.textContent = 'No image selected';
            this.updateProcessButton();
            return;
        }

        try {
            this.showLoading();
            const response = await fetch(`${this.baseURL}/api/demo-images/${filename}`);
            
            if (!response.ok) {
                throw new Error('Failed to load demo image');
            }
            
            const result = await response.json();
            
            // Set current image data
            this.currentImage = result.image;
            this.currentImageDimensions = { width: result.width, height: result.height };
            
            // Clear file input when demo is selected
            this.elements.imageInput.value = '';
            
            // Show preview and update status
            this.elements.demoPreviewImg.src = result.image;
            this.elements.demoPreview.style.display = 'block';
            this.elements.uploadStatus.textContent = `${filename} (${result.width}x${result.height} → ${result.width/4}x${result.height/4})`;
            
            // Update process button
            this.updateProcessButton();
            this.hideError();
            
        } catch (error) {
            this.showError(`Failed to load demo image: ${error.message}`);
            this.currentImage = null;
            this.currentImageDimensions = null;
            this.elements.demoPreview.style.display = 'none';
            this.elements.uploadStatus.textContent = 'No image selected';
        } finally {
            this.hideLoading();
        }
    }
    
    async processImage() {
        if (!this.currentImage) return;
        
        const downsampler = this.elements.downsamplerSelect.value;
        const method = this.elements.methodSelect.value;
        const targetResolution = this.currentImageDimensions.width / 4;
        
        this.showLoading();
        this.hideError();
        this.hideResults();
        
        try {
            const response = await fetch(`${this.baseURL}/api/downsample`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: this.currentImage,
                    downsampler: downsampler,
                    method: method,
                    target_resolution: targetResolution
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Processing failed');
            }
            
            const result = await response.json();
            this.showResults(result);
            
        } catch (error) {
            this.showError(`Processing failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    showResults(result) {
        // Update metadata
        this.elements.metadata.innerHTML = `
            <strong>${result.downsampler}</strong> - ${result.method}<br>
            Original: ${result.original_size} → Downsampled: ${result.downsampled_size}
        `;
        
        // Update images
        this.elements.originalImage.src = result.original;
        this.elements.downsampledImage.src = result.downsampled;
        
        // Update labels
        document.querySelector('.downsampled-label').textContent = `Downsampled (${result.downsampled_size})`;
        document.querySelector('.original-label').textContent = `Original (${result.original_size})`;
        
        // Reset comparison slider
        this.elements.comparisonSlider.value = 50;
        this.updateComparison({ target: { value: 50 } });
        
        // Add click event to image container for direct positioning
        setTimeout(() => {
            const imageContainer = document.querySelector('.image-container');
            if (imageContainer) {
                imageContainer.addEventListener('click', (e) => {
                    const rect = imageContainer.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
                    
                    this.elements.comparisonSlider.value = percentage;
                    this.updateComparison({ target: { value: percentage } });
                });
            }
        }, 100);
        
        // Show results
        this.elements.results.style.display = 'block';
    }
    
    updateComparison(event) {
        const value = event.target.value;
        const percentage = 100 - value; // Invert the value so 0 shows original, 100 shows downsampled
        
        // Update downsampled image clip-path
        this.elements.downsampledImage.style.clipPath = `inset(0 ${percentage}% 0 0)`;
        
        // Update slider handle position
        this.elements.sliderHandle.style.left = `${value}%`;
    }
    
    showLoading() {
        this.elements.loading.style.display = 'block';
    }
    
    hideLoading() {
        this.elements.loading.style.display = 'none';
    }
    
    hideResults() {
        this.elements.results.style.display = 'none';
    }
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.error.style.display = 'block';
    }
    
    hideError() {
        this.elements.error.style.display = 'none';
    }

    // Tab switching
    switchTab(tabName) {
        this.currentTab = tabName;
        
        // Update nav tabs
        this.elements.navTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        // Update tab content
        this.elements.tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        // Hide results when switching tabs
        this.hideResults();
        this.hideAdversarialResults();
    }

    // Load available decoy images
    async loadDecoyImages() {
        try {
            const response = await fetch(`${this.baseURL}/api/decoy-images`);
            this.decoyImages = await response.json();
            this.populateDecoyImages();
        } catch (error) {
            console.error('Failed to load decoy images:', error);
        }
    }

    populateDecoyImages() {
        this.elements.decoySelect.innerHTML = '<option value="">Select decoy image...</option>';
        
        this.decoyImages.forEach(decoy => {
            const option = document.createElement('option');
            option.value = decoy.filename;
            option.textContent = `${decoy.filename} (${decoy.width}x${decoy.height})`;
            this.elements.decoySelect.appendChild(option);
        });
    }

    // Generate text preview
    async generateTextPreview() {
        const text = this.elements.targetText.value.trim();
        if (!text) {
            this.showError('Please enter text to generate preview');
            return;
        }

        try {
            this.showLoading();
            
            const fontSize = parseInt(this.elements.fontSizeInput.value);
            const alignment = this.elements.textAlignInput.value;
            
            const response = await fetch(`${this.baseURL}/api/generate-text-image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    size: 1092, // Default preview size
                    font_size: fontSize,
                    alignment: alignment
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate text preview');
            }

            const result = await response.json();
            this.currentTextPreview = result.image;
            this.elements.textPreviewImg.src = result.image;
            this.elements.textPreview.style.display = 'block';
            
            // Show warning if text overflowed
            if (result.text_overflowed) {
                this.elements.textWarning.textContent = 'Warning: Text is too large and may be cut off in the image.';
                this.elements.textWarning.style.display = 'block';
                this.elements.textWarning.style.color = '#ff6b6b';
            } else {
                this.elements.textWarning.style.display = 'none';
            }
            
            this.updateGenerateAdvButton();

        } catch (error) {
            this.showError(`Failed to generate text preview: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    // Handle method change for adversarial generation
    handleMethodChange(event) {
        const method = event.target.value;
        
        // Hide all method-specific parameters first
        this.elements.bicubicParams.style.display = 'none';
        this.elements.bilinearParams.style.display = 'none';
        this.elements.nearestParams.style.display = 'none';
        
        // Update default values based on method
        if (method === 'bicubic') {
            this.elements.bicubicParams.style.display = 'block';
            this.elements.lamInput.value = '0.25';
            this.elements.epsInput.value = '0.0';
            this.elements.gammaInput.value = '1.0';
        } else if (method === 'bilinear') {
            this.elements.bilinearParams.style.display = 'block';
            this.elements.lamInput.value = '1.0';
            this.elements.epsInput.value = '0.0';
            this.elements.gammaInput.value = '0.9';
        } else if (method === 'nearest') {
            this.elements.nearestParams.style.display = 'block';
            this.elements.lamInput.value = '0.25';
            this.elements.epsInput.value = '0.0';
            this.elements.gammaInput.value = '1.0';
        }
    }

    // Update generate adversarial button state
    updateGenerateAdvButton() {
        const hasText = this.elements.targetText.value.trim() !== '';
        const hasDecoy = this.elements.decoySelect.value !== '';
        
        this.elements.generateAdvBtn.disabled = !(hasText && hasDecoy);
    }

    // Generate adversarial image
    async generateAdversarial() {
        const text = this.elements.targetText.value.trim();
        const decoyFilename = this.elements.decoySelect.value;
        
        if (!text || !decoyFilename) {
            this.showError('Please enter text and select a decoy image');
            return;
        }

        try {
            this.showLoading();
            this.hideError();
            this.hideAdversarialResults();
            
            const method = this.elements.methodSelect2.value;
            const requestData = {
                method: method,
                text: text,
                decoy_filename: decoyFilename,
                lam: parseFloat(this.elements.lamInput.value),
                eps: parseFloat(this.elements.epsInput.value),
                gamma: parseFloat(this.elements.gammaInput.value)
            };

            // Add method-specific parameters
            if (method === 'bicubic') {
                requestData.dark_frac = parseFloat(this.elements.darkFracInput.value);
            } else if (method === 'bilinear') {
                requestData.dark_frac = parseFloat(this.elements.darkFracInputBilinear.value);
            } else if (method === 'nearest') {
                requestData.offset = parseInt(this.elements.offsetInput.value);
            }

            const response = await fetch(`${this.baseURL}/api/generate-adversarial`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate adversarial image');
            }

            const result = await response.json();
            this.showAdversarialResults(result);

        } catch (error) {
            this.showError(`Failed to generate adversarial image: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    // Show adversarial results
    showAdversarialResults(result) {
        // Update metadata
        const params = result.parameters;
        let paramStr = `λ=${params.lam}, ε=${params.eps}, γ=${params.gamma}`;
        if (params.offset !== null) paramStr += `, offset=${params.offset}`;
        if (params.dark_frac !== null) paramStr += `, dark_frac=${params.dark_frac}`;
        
        this.elements.advMetadata.innerHTML = `
            <strong>Method:</strong> ${result.method}<br>
            <strong>Text:</strong> "${result.text}"<br>
            <strong>Decoy:</strong> ${result.decoy_filename}<br>
            <strong>Parameters:</strong> ${paramStr}
        `;

        // Update images
        this.elements.originalDecoy.src = result.original_decoy;
        this.elements.targetImage.src = result.target_image;
        this.elements.adversarialImage.src = result.adversarial_image;

        // Show results
        this.elements.adversarialResults.style.display = 'block';
    }

    // Hide adversarial results
    hideAdversarialResults() {
        this.elements.adversarialResults.style.display = 'none';
    }

    // Hide text preview when settings change
    hideTextPreview() {
        this.elements.textPreview.style.display = 'none';
        this.elements.textWarning.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnamorpherApp();
});