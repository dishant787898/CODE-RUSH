document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS
    AOS.init({
        duration: 800,
        easing: 'ease-out',
        once: true
    });
    
    // Initialize particles background
    if (document.querySelector('.particles') && window.particlesJS) {
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
        });
    }
    
    // NFT Marketplace Filtering
    const filterBtns = document.querySelectorAll('.filter-btn');
    const nftCards = document.querySelectorAll('.nft-card');
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            filterBtns.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            btn.classList.add('active');
            
            // Get filter value
            const filterValue = btn.getAttribute('data-filter');
            
            // Filter cards
            nftCards.forEach(card => {
                const category = card.getAttribute('data-category');
                
                if (filterValue === 'all' || filterValue === category) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 100);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });
    
    // Modal Functionality
    const modal = document.getElementById('nft-modal');
    const buyBtns = document.querySelectorAll('.btn-buy-nft');
    const closeModalBtns = document.querySelectorAll('.close-modal');
    const confirmBtn = document.getElementById('confirm-purchase');
    
    // Buy buttons open modal
    buyBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Get NFT card info
            const card = this.closest('.nft-card');
            const nftId = this.getAttribute('data-id');
            const nftImage = card.querySelector('.nft-image img').src;
            const nftName = card.querySelector('h3').textContent;
            const nftDesc = card.querySelector('.nft-description').textContent;
            const nftPrice = card.querySelector('.nft-price').textContent.trim();
            
            // Populate modal
            document.getElementById('modal-item-image').src = nftImage;
            document.getElementById('modal-item-name').textContent = nftName;
            document.getElementById('modal-item-description').textContent = nftDesc;
            document.getElementById('modal-item-price').textContent = nftPrice.replace(/[^\d.]/g, '');
            
            // Show modal
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Close modal functionality
    closeModalBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            modal.classList.remove('show');
            document.body.style.overflow = 'auto';
        });
    });
    
    // Click outside modal closes it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.classList.remove('show');
            document.body.style.overflow = 'auto';
        }
    });
    
    // Confirm purchase button
    if (confirmBtn) {
        confirmBtn.addEventListener('click', function() {
            const walletAddress = document.getElementById('wallet-address').value;
            
            if (!walletAddress) {
                alert('Please enter your wallet address');
                return;
            }
            
            // Show loading overlay
            const loadingOverlay = document.getElementById('loading-overlay');
            loadingOverlay.classList.add('active');
            
            // Simulate transaction process
            setTimeout(() => {
                loadingOverlay.classList.remove('active');
                modal.classList.remove('show');
                document.body.style.overflow = 'auto';
                
                // Show success notification
                showNotification('NFT purchased successfully! Check your wallet in a few minutes.', 'success');
            }, 2000);
        });
    }
    
    // Testimonial Slider
    const testimonials = document.querySelectorAll('.testimonial-card');
    const indicators = document.querySelectorAll('.indicator');
    const prevBtn = document.querySelector('.prev-testimonial');
    const nextBtn = document.querySelector('.next-testimonial');
    let currentTestimonial = 0;
    
    function showTestimonial(index) {
        // Hide all testimonials
        testimonials.forEach(testimonial => {
            testimonial.classList.remove('active');
        });
        
        // Remove active class from all indicators
        indicators.forEach(indicator => {
            indicator.classList.remove('active');
        });
        
        // Show current testimonial and indicator
        testimonials[index].classList.add('active');
        indicators[index].classList.add('active');
    }
    
    // Next button
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            currentTestimonial = (currentTestimonial + 1) % testimonials.length;
            showTestimonial(currentTestimonial);
        });
    }
    
    // Previous button
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            currentTestimonial = (currentTestimonial - 1 + testimonials.length) % testimonials.length;
            showTestimonial(currentTestimonial);
        });
    }
    
    // Indicators
    indicators.forEach((indicator, index) => {
        indicator.addEventListener('click', function() {
            currentTestimonial = index;
            showTestimonial(currentTestimonial);
        });
    });
    
    // Auto rotate testimonials
    setInterval(() => {
        if (testimonials.length > 0) {
            currentTestimonial = (currentTestimonial + 1) % testimonials.length;
            showTestimonial(currentTestimonial);
        }
    }, 8000);
    
    // Create NFT button
    const createNftBtn = document.querySelector('.btn-create-nft');
    if (createNftBtn) {
        createNftBtn.addEventListener('click', function() {
            // In a real application, this would navigate to a create NFT page
            showNotification('NFT creation feature will be available soon!', 'info');
        });
    }
    
    // Subscribe buttons
    const subscribeButtons = document.querySelectorAll('.btn-subscribe');
    subscribeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const featureName = this.closest('.premium-card').querySelector('h3').textContent;
            showNotification(`You've subscribed to ${featureName}!`, 'success');
        });
    });
    
    // Contact sales button
    const contactButton = document.querySelector('.btn-contact');
    if (contactButton) {
        contactButton.addEventListener('click', function() {
            // Smooth scroll to contact form
            document.getElementById('contact').scrollIntoView({
                behavior: 'smooth'
            });
        });
    }
    
    // Contact form submission
    const contactForm = document.querySelector('.contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading overlay
            const loadingOverlay = document.getElementById('loading-overlay');
            loadingOverlay.classList.add('active');
            
            // Simulate form submission
            setTimeout(() => {
                loadingOverlay.classList.remove('active');
                
                // Clear form
                this.reset();
                
                // Show success notification
                showNotification('Your message has been sent! Our team will contact you shortly.', 'success');
            }, 1500);
        });
    }
    
    // Notification function
    function showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        // Create icon based on type
        let icon = 'fa-info-circle';
        if (type === 'success') icon = 'fa-check-circle';
        if (type === 'error') icon = 'fa-exclamation-circle';
        
        notification.innerHTML = `
            <i class="fas ${icon}"></i>
            <p>${message}</p>
            <span class="close-notification">&times;</span>
        `;
        
        // Add to document
        document.body.appendChild(notification);
        
        // Show with animation
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto close after 5 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 5000);
        
        // Close button functionality
        const closeBtn = notification.querySelector('.close-notification');
        closeBtn.addEventListener('click', function() {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                notification.remove();
            }, 300);
        });
    }
    
    // Add notification styles if not already in main CSS
    if (!document.querySelector('style.notification-styles')) {
        const style = document.createElement('style');
        style.className = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 15px 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                transform: translateX(100%);
                transition: transform 0.3s ease;
                z-index: 1000;
                min-width: 300px;
                max-width: 500px;
            }
            .notification i {
                font-size: 1.5rem;
                margin-right: 15px;
            }
            .notification p {
                flex: 1;
                margin: 0;
            }
            .notification .close-notification {
                cursor: pointer;
                font-size: 1.2rem;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }
            .notification .close-notification:hover {
                opacity: 1;
            }
            .notification.success {
                border-left: 4px solid #2ecc71;
            }
            .notification.success i {
                color: #2ecc71;
            }
            .notification.error {
                border-left: 4px solid #e74c3c;
            }
            .notification.error i {
                color: #e74c3c;
            }
            .notification.info {
                border-left: 4px solid #3498db;
            }
            .notification.info i {
                color: #3498db;
            }
            [data-theme="dark"] .notification {
                background: #1e1e1e;
                color: #f5f5f5;
            }
        `;
        document.head.appendChild(style);
    }
});
