document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS
    AOS.init({
        duration: 800,
        easing: 'ease-out',
        once: true
    });
    
    // Theme switcher
    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', savedTheme);
        themeSwitch.checked = savedTheme === 'dark';
        
        // Theme switch event listener
        themeSwitch.addEventListener('change', function() {
            const theme = this.checked ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            
            // Redraw charts
            updateChartsForTheme();
        });
    }
    
    // Function to get chart text color based on theme
    function getChartTextColor() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? '#ecf0f1' : '#2c3e50';
    }
    
    // Function to update all charts for current theme
    function updateChartsForTheme() {
        const textColor = getChartTextColor();
        
        if (window.distributionChart) {
            window.distributionChart.options.plugins.legend.labels.color = textColor;
            window.distributionChart.options.scales.y.ticks.color = textColor;
            window.distributionChart.options.scales.x.ticks.color = textColor;
            window.distributionChart.update();
        }
        
        if (window.modelUsageChart) {
            window.modelUsageChart.options.plugins.legend.labels.color = textColor;
            window.modelUsageChart.options.scales.y.ticks.color = textColor;
            window.modelUsageChart.options.scales.x.ticks.color = textColor;
            window.modelUsageChart.update();
        }
        
        if (window.confidenceChart) {
            window.confidenceChart.options.plugins.legend.labels.color = textColor;
            window.confidenceChart.options.scales.y.ticks.color = textColor;
            window.confidenceChart.options.scales.x.ticks.color = textColor;
            window.confidenceChart.update();
        }
    }
    
    // Initialize charts
    const textColor = getChartTextColor();
    
    // Distribution Chart
    const ctxDistribution = document.getElementById('distributionChart');
    if (ctxDistribution) {
        window.distributionChart = new Chart(ctxDistribution, {
            type: 'doughnut',
            data: {
                labels: ['Recyclable', 'Organic'],
                datasets: [{
                    data: [63, 37], // Sample data - will be replaced with actual data from backend
                    backgroundColor: [
                        '#3498db',
                        '#2ecc71'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: textColor
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Model Usage Chart
    const ctxModelUsage = document.getElementById('modelUsageChart');
    if (ctxModelUsage) {
        window.modelUsageChart = new Chart(ctxModelUsage, {
            type: 'bar',
            data: {
                labels: ['VGG16', 'ResNet50', 'InceptionV3', 'Ensemble'],
                datasets: [{
                    label: 'Usage Count',
                    data: [42, 28, 36, 85], // Sample data
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(44, 62, 80, 0.7)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(44, 62, 80, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: textColor
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        title: {
                            display: true,
                            text: 'Number of Classifications',
                            color: textColor
                        }
                    },
                    x: {
                        ticks: {
                            color: textColor
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }
            }
        });
    }
    
    // Confidence Trends Chart
    const ctxConfidence = document.getElementById('confidenceChart');
    if (ctxConfidence) {
        window.confidenceChart = new Chart(ctxConfidence, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [
                    {
                        label: 'VGG16',
                        data: [92, 93, 94, 95, 95, 96],
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'ResNet50',
                        data: [91, 92, 93, 94, 95, 95],
                        borderColor: 'rgba(46, 204, 113, 1)',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'InceptionV3',
                        data: [93, 94, 95, 96, 96, 97],
                        borderColor: 'rgba(155, 89, 182, 1)',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Ensemble',
                        data: [95, 96, 97, 97, 98, 98],
                        borderColor: 'rgba(44, 62, 80, 1)',
                        backgroundColor: 'rgba(44, 62, 80, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: textColor
                        }
                    }
                },
                scales: {
                    y: {
                        min: 85,
                        max: 100,
                        ticks: {
                            color: textColor,
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Average Confidence',
                            color: textColor
                        }
                    },
                    x: {
                        ticks: {
                            color: textColor
                        }
                    }
                }
            }
        });
    }
    
    // Compare models form handling
    const compareForm = document.getElementById('compare-form');
    const compareFileInput = document.getElementById('compare-file-input');
    const compareFileName = document.getElementById('compare-file-name');
    
    if (compareFileInput) {
        compareFileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                compareFileName.textContent = this.files[0].name;
            } else {
                compareFileName.textContent = 'Choose an image to compare';
            }
        });
    }
    
    if (compareForm) {
        compareForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            // Show loading state
            document.getElementById('comparison-results').style.display = 'none';
            const btn = document.querySelector('.btn-compare');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            // Send to backend for processing
            fetch('/compare_models', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }
                
                // Display image
                document.getElementById('comparison-img').src = '/static/' + data.image_path;
                
                // Update VGG16 result
                updateModelResult('vgg', data.vgg16_class, data.vgg16_confidence);
                
                // Update ResNet50 result
                updateModelResult('resnet', data.resnet50_class, data.resnet50_confidence);
                
                // Update InceptionV3 result
                updateModelResult('inception', data.inception_class, data.inception_confidence);
                
                // Update Ensemble result
                updateModelResult('ensemble', data.ensemble_class, data.ensemble_confidence);
                
                // Update conclusion
                document.getElementById('comparison-text').textContent = data.conclusion;
                
                // Show results
                document.getElementById('comparison-results').style.display = 'block';
                
                // Reset button
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-sync-alt"></i> Run Comparison';
            })
            .catch(error => {
                console.error('Error:', error);
                alert("An error occurred during the comparison.");
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-sync-alt"></i> Run Comparison';
            });
        });
    }
    
    function updateModelResult(modelClass, className, confidence) {
        const modelElement = document.querySelector(`.model-result.${modelClass}`);
        const resultPill = modelElement.querySelector('.result-pill');
        const confidenceFill = modelElement.querySelector('.confidence-fill');
        const confidenceText = modelElement.querySelector('.confidence-text');
        
        // Set class name
        resultPill.textContent = className;
        resultPill.className = 'result-pill ' + (className.includes('Recyclable') ? 'recyclable' : 'organic');
        
        // Set confidence
        const confidenceValue = parseFloat(confidence) * 100;
        confidenceFill.style.width = `${confidenceValue}%`;
        confidenceText.textContent = `${confidenceValue.toFixed(1)}%`;
    }
});
