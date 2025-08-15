// Theme toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS
    AOS.init({
        duration: 800,
        easing: 'ease-out',
        once: true
    });

    // File input handling
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // Update file input handling for better feedback
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                fileName.textContent = file.name;
                
                // Create success notification
                const uploadFeedback = document.createElement('div');
                uploadFeedback.className = 'upload-feedback success';
                uploadFeedback.innerHTML = `
                    <i class="fas fa-check-circle"></i>
                    <span>"${file.name}" successfully uploaded</span>
                    <div class="upload-progress-bar">
                        <div class="upload-progress"></div>
                    </div>
                `;
                
                // Remove any existing feedback
                const existingFeedback = document.querySelector('.upload-feedback');
                if (existingFeedback) {
                    existingFeedback.remove();
                }
                
                // Add the feedback below file input
                const uploadContainer = document.querySelector('.file-input-container');
                uploadContainer.appendChild(uploadFeedback);
                
                // Animate progress bar
                setTimeout(() => {
                    const progressBar = document.querySelector('.upload-progress');
                    progressBar.style.width = '100%';
                }, 100);
                
                // Preview the image
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewImg = document.createElement('img');
                    previewImg.src = e.target.result;
                    previewImg.className = 'file-preview';
                    
                    // Remove any existing preview
                    const existingPreview = document.querySelector('.file-preview');
                    if (existingPreview) {
                        existingPreview.remove();
                    }
                    
                    // Add new preview after the upload box
                    const uploadBox = document.querySelector('.file-upload-box');
                    uploadBox.parentNode.appendChild(previewImg);
                }
                reader.readAsDataURL(file);
            } else {
                fileName.textContent = 'Choose an image';
                
                // Remove feedback and preview if no file selected
                const existingFeedback = document.querySelector('.upload-feedback');
                if (existingFeedback) {
                    existingFeedback.remove();
                }
                
                const existingPreview = document.querySelector('.file-preview');
                if (existingPreview) {
                    existingPreview.remove();
                }
            }
        });
    }
    
    // Show loading overlay when form is submitted
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            loadingOverlay.classList.add('active');
        });
    }
    
    // Initialize tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    if (tabButtons.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab') + '-tab';
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // Animate counting for stat numbers
    const statNumbers = document.querySelectorAll('.stat-number');
    if (statNumbers.length > 0) {
        const observerOptions = {
            threshold: 0.5
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const number = entry.target;
                    const countTo = parseInt(number.dataset.count) || 0;
                    let count = 0;
                    const duration = 2000;
                    const increment = countTo / (duration / 30);
                    
                    const interval = setInterval(() => {
                        count += increment;
                        if (count >= countTo) {
                            count = countTo;
                            clearInterval(interval);
                        }
                        number.textContent = Math.floor(count);
                    }, 30);
                    
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);
        
        statNumbers.forEach(number => {
            observer.observe(number);
        });
    }
    
    // Theme switcher
    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
        // Check for saved theme preference or use preferred color scheme
        const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        
        // Apply saved theme
        document.documentElement.setAttribute('data-theme', savedTheme);
        themeSwitch.checked = savedTheme === 'dark';
        
        // Theme switch event listener
        themeSwitch.addEventListener('change', function() {
            const theme = this.checked ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            
            // Redraw charts if they exist
            if (window.accuracyChart) {
                updateChartTheme(window.accuracyChart);
            }
            if (window.comparisonChart) {
                updateChartTheme(window.comparisonChart);
            }
            if (window.historyChart) {
                updateChartTheme(window.historyChart);
            }
        });
    }
    
    // Initialize particles for hero section
    const particlesContainer = document.querySelector('.particles');
    if (particlesContainer && window.particlesJS) {
        particlesJS('particles', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: "#ffffff" },
                shape: { type: "circle" },
                opacity: { value: 0.5, random: false },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: "#ffffff",
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: "none",
                    random: false,
                    straight: false,
                    out_mode: "out",
                    bounce: false
                }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: { enable: true, mode: "repulse" },
                    onclick: { enable: true, mode: "push" },
                    resize: true
                }
            },
            retina_detect: true
        });
    }
    
    // Add animation to progress bars
    const progressBars = document.querySelectorAll('.progress');
    progressBars.forEach(bar => {
        setTimeout(() => {
            const width = bar.style.width;
            bar.style.width = "0";
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        }, 500);
    });
    
    // Add animation to metric bars
    const metricBars = document.querySelectorAll('.bar-fill');
    if (metricBars.length > 0) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const bar = entry.target;
                    const width = bar.style.width;
                    bar.style.width = "0";
                    setTimeout(() => {
                        bar.style.width = width;
                    }, 100);
                    observer.unobserve(bar);
                }
            });
        }, { threshold: 0.5 });
        
        metricBars.forEach(bar => {
            observer.observe(bar);
        });
    }
    
    // Check if we're on the business page
    if (window.location.pathname === '/business') {
        // Make sure business.js is loaded
        if (!document.querySelector('script[src*="business.js"]')) {
            const businessScript = document.createElement('script');
            businessScript.src = '/static/js/business.js';
            document.body.appendChild(businessScript);
        }
    }
});

// Initialize all charts
function initCharts() {
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDarkMode ? '#ecf0f1' : '#2c3e50';
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Common chart options
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: textColor
                }
            }
        },
        scales: {
            y: {
                grid: {
                    color: gridColor
                },
                ticks: {
                    color: textColor
                }
            },
            x: {
                grid: {
                    color: gridColor
                },
                ticks: {
                    color: textColor
                }
            }
        },
        animation: {
            duration: 2000,
            easing: 'easeOutQuart'
        }
    };
    
    // Metrics Chart - Using actual data from the uploaded image
    const ctxMetrics = document.getElementById('metricsChart')?.getContext('2d');
    if (ctxMetrics) {
        window.metricsChart = new Chart(ctxMetrics, {
            type: 'bar',
            data: {
                labels: ['VGG16', 'ResNet50', 'InceptionV3', 'Ensemble'],
                datasets: [
                    {
                        label: 'Accuracy',
                        data: [96.50, 96.00, 97.00, 98.00],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Precision',
                        data: [96.15, 95.65, 96.65, 97.65],
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(46, 204, 113, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Recall',
                        data: [96.75, 96.25, 97.25, 98.25],
                        backgroundColor: 'rgba(155, 89, 182, 0.7)',
                        borderColor: 'rgba(155, 89, 182, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'F1-Score',
                        data: [96.45, 96.00, 96.95, 97.95],
                        backgroundColor: 'rgba(241, 196, 15, 0.7)',
                        borderColor: 'rgba(241, 196, 15, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                ...chartOptions,
                scales: {
                    y: {
                        ...chartOptions.scales.y,
                        beginAtZero: false,
                        min: 94,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Score (%)',
                            color: textColor
                        }
                    }
                },
                plugins: {
                    ...chartOptions.plugins,
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Accuracy Chart
    const ctxAccuracy = document.getElementById('accuracyChart').getContext('2d');
    window.accuracyChart = new Chart(ctxAccuracy, {
        type: 'bar',
        data: {
            labels: ['VGG16', 'ResNet50', 'InceptionV3', 'Ensemble'],
            datasets: [{
                label: 'Model Accuracy (%)',
                data: [96.4, 97.2, 96.8, 98.5],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.6)',
                    'rgba(46, 204, 113, 0.6)',
                    'rgba(155, 89, 182, 0.6)',
                    'rgba(52, 73, 94, 0.6)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(155, 89, 182, 1)',
                    'rgba(52, 73, 94, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                y: {
                    ...chartOptions.scales.y,
                    beginAtZero: false,
                    min: 90,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: textColor
                    }
                }
            },
            plugins: {
                ...chartOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Accuracy: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Comparison Chart
    const ctxComparison = document.getElementById('comparisonChart').getContext('2d');
    window.comparisonChart = new Chart(ctxComparison, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Speed'],
            datasets: [
                {
                    label: 'VGG16',
                    data: [96.4, 95.8, 97.0, 96.4, 85],
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(52, 152, 219, 1)'
                },
                {
                    label: 'ResNet50',
                    data: [97.2, 96.9, 97.5, 97.2, 90],
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    borderColor: 'rgba(46, 204, 113, 1)',
                    pointBackgroundColor: 'rgba(46, 204, 113, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(46, 204, 113, 1)'
                },
                {
                    label: 'InceptionV3',
                    data: [96.8, 97.0, 96.5, 96.7, 88],
                    backgroundColor: 'rgba(155, 89, 182, 0.2)',
                    borderColor: 'rgba(155, 89, 182, 1)',
                    pointBackgroundColor: 'rgba(155, 89, 182, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(155, 89, 182, 1)'
                },
                {
                    label: 'Ensemble',
                    data: [98.5, 98.2, 98.7, 98.5, 80],
                    backgroundColor: 'rgba(52, 73, 94, 0.2)',
                    borderColor: 'rgba(52, 73, 94, 1)',
                    pointBackgroundColor: 'rgba(52, 73, 94, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(52, 73, 94, 1)'
                }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                r: {
                    min: 75,
                    max: 100,
                    ticks: {
                        color: textColor,
                        backdropColor: 'transparent'
                    },
                    grid: {
                        color: gridColor
                    },
                    angleLines: {
                        color: gridColor
                    },
                    pointLabels: {
                        color: textColor
                    }
                }
            }
        }
    });
    
    // History Chart
    const ctxHistory = document.getElementById('historyChart').getContext('2d');
    window.historyChart = new Chart(ctxHistory, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
            datasets: [
                {
                    label: 'VGG16',
                    data: [92.1, 93.5, 94.2, 94.8, 95.1, 95.4, 95.7, 96.0, 96.2, 96.4],
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'ResNet50',
                    data: [93.2, 94.3, 95.0, 95.5, 96.0, 96.3, 96.6, 96.8, 97.0, 97.2],
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderColor: 'rgba(46, 204, 113, 1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'InceptionV3',
                    data: [92.8, 93.9, 94.5, 95.1, 95.5, 95.9, 96.2, 96.4, 96.6, 96.8],
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderColor: 'rgba(155, 89, 182, 1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Ensemble',
                    data: [94.5, 95.4, 96.2, 96.7, 97.1, 97.5, 97.8, 98.0, 98.3, 98.5],
                    backgroundColor: 'rgba(52, 73, 94, 0.1)',
                    borderColor: 'rgba(52, 73, 94, 1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                y: {
                    ...chartOptions.scales.y,
                    beginAtZero: false,
                    min: 90,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: textColor
                    }
                },
                x: {
                    ...chartOptions.scales.x,
                    title: {
                        display: true,
                        text: 'Month',
                        color: textColor
                    }
                }
            }
        }
    });
}

// Update chart theme based on current theme
function updateChartTheme(chart) {
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDarkMode ? '#ecf0f1' : '#2c3e50';
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Update scale colors
    if (chart.options.scales.r) {
        // Radar chart
        chart.options.scales.r.ticks.color = textColor;
        chart.options.scales.r.grid.color = gridColor;
        chart.options.scales.r.angleLines.color = gridColor;
        chart.options.scales.r.pointLabels.color = textColor;
    } else {
        // Other charts
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.y.grid.color = gridColor;
        chart.options.scales.y.title.color = textColor;
        
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.x.grid.color = gridColor;
        if (chart.options.scales.x.title) {
            chart.options.scales.x.title.color = textColor;
        }
    }
    
    // Update legend colors
    chart.options.plugins.legend.labels.color = textColor;
    
    chart.update();
}

// Animate counting up for stat numbers
function animateStatNumbers() {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    if (statNumbers.length) {
        statNumbers.forEach(statNumber => {
            const target = parseInt(statNumber.getAttribute('data-count'));
            const duration = 2000;
            let start = 0;
            const startTime = performance.now();
            
            function updateNumber(currentTime) {
                const elapsedTime = currentTime - startTime;
                const progress = Math.min(elapsedTime / duration, 1);
                const easeProgress = easeOutQuad(progress);
                const current = Math.floor(easeProgress * target);
                
                statNumber.textContent = current;
                
                if (progress < 1) {
                    requestAnimationFrame(updateNumber);
                } else {
                    statNumber.textContent = target;
                }
            }
            
            requestAnimationFrame(updateNumber);
        });
    }
}

// Easing function for smoother animations
function easeOutQuad(t) {
    return t * (2 - t);
}

// Add ripple effect to buttons
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('btn-classify') || e.target.closest('.btn-classify')) {
        const button = e.target.classList.contains('btn-classify') ? e.target : e.target.closest('.btn-classify');
        
        const circle = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        
        circle.style.width = circle.style.height = `${diameter}px`;
        
        const rect = button.getBoundingClientRect();
        
        circle.style.left = `${e.clientX - rect.left - diameter / 2}px`;
        circle.style.top = `${e.clientY - rect.top - diameter / 2}px`;
        
        circle.classList.add('ripple');
        
        const ripple = button.getElementsByClassName('ripple')[0];
        
        if (ripple) {
            ripple.remove();
        }
        
        button.appendChild(circle);
    }
});

// Show custom file name on mobile devices
window.addEventListener('resize', function() {
    const fileName = document.getElementById('file-name');
    if (fileName && window.innerWidth < 768) {
        const originalText = fileName.textContent;
        if (originalText !== 'Choose an image' && originalText.length > 15) {
            fileName.textContent = originalText.substring(0, 12) + '...';
        }
    }
});
