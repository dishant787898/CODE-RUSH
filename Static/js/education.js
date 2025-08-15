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
            if (window.wasteCompositionChart) {
                updateChartTheme(window.wasteCompositionChart);
            }
        });
    }
    
    // Waste Composition Chart
    const ctxWaste = document.getElementById('wasteCompositionChart');
    if (ctxWaste) {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#ecf0f1' : '#2c3e50';
        
        window.wasteCompositionChart = new Chart(ctxWaste, {
            type: 'pie',
            data: {
                labels: ['Food & Green', 'Paper & Cardboard', 'Plastic', 'Glass', 'Metal', 'Other'],
                datasets: [{
                    data: [44, 17, 12, 5, 4, 18],
                    backgroundColor: [
                        '#2ecc71',
                        '#3498db',
                        '#e74c3c',
                        '#9b59b6',
                        '#f39c12',
                        '#95a5a6'
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
                            color: textColor,
                            font: {
                                size: 12
                            }
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
    
    function updateChartTheme(chart) {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#ecf0f1' : '#2c3e50';
        
        if (chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = textColor;
        }
        
        chart.update();
    }
    
    // FAQ Accordion
    const faqItems = document.querySelectorAll('.faq-item');
    if (faqItems.length > 0) {
        faqItems.forEach(item => {
            const header = item.querySelector('.faq-header');
            
            header.addEventListener('click', () => {
                // Close all other items
                faqItems.forEach(otherItem => {
                    if (otherItem !== item && otherItem.classList.contains('active')) {
                        otherItem.classList.remove('active');
                        otherItem.querySelector('.faq-content').style.maxHeight = null;
                    }
                });
                
                // Toggle current item
                item.classList.toggle('active');
                const content = item.querySelector('.faq-content');
                
                if (item.classList.contains('active')) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                } else {
                    content.style.maxHeight = null;
                }
            });
        });
    }
    
    // Interactive waste quiz
    const quizForm = document.getElementById('waste-quiz-form');
    if (quizForm) {
        quizForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get all quiz questions
            const questions = document.querySelectorAll('.quiz-question');
            let score = 0;
            let totalQuestions = questions.length;
            
            // Check each question
            questions.forEach(question => {
                const questionId = question.getAttribute('data-question-id');
                const selectedOption = document.querySelector(`input[name="q${questionId}"]:checked`);
                const correctAnswer = question.getAttribute('data-correct');
                
                if (selectedOption) {
                    // Mark answers
                    if (selectedOption.value === correctAnswer) {
                        score++;
                        question.classList.add('correct');
                    } else {
                        question.classList.add('incorrect');
                    }
                }
                
                // Show correct answer
                const options = question.querySelectorAll('.quiz-option');
                options.forEach(option => {
                    const input = option.querySelector('input');
                    if (input.value === correctAnswer) {
                        option.classList.add('correct-answer');
                    }
                });
            });
            
            // Show result
            const resultElement = document.getElementById('quiz-result');
            if (resultElement) {
                resultElement.textContent = `Your score: ${score}/${totalQuestions}`;
                resultElement.style.display = 'block';
                
                // Scroll to results
                resultElement.scrollIntoView({ behavior: 'smooth' });
            }
        });
        
        // Reset quiz
        const resetButton = document.getElementById('reset-quiz');
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                // Clear selections
                const questions = document.querySelectorAll('.quiz-question');
                questions.forEach(question => {
                    question.classList.remove('correct', 'incorrect');
                    
                    const options = question.querySelectorAll('.quiz-option');
                    options.forEach(option => {
                        option.classList.remove('correct-answer');
                        const input = option.querySelector('input');
                        input.checked = false;
                    });
                });
                
                // Hide result
                const resultElement = document.getElementById('quiz-result');
                if (resultElement) {
                    resultElement.style.display = 'none';
                }
            });
        }
    }
});
