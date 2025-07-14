// Main JavaScript file for the blog

document.addEventListener('DOMContentLoaded', function() {
    // Copy to clipboard functionality
    initializeCopyButtons();
    
    // Table of contents generation
    generateTableOfContents();
    
    // Reading time estimation
    calculateReadingTime();
    
    // Smooth scrolling for anchor links
    initializeSmoothScrolling();
});

// Copy to clipboard functionality for code blocks
function initializeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(function(codeBlock) {
        const pre = codeBlock.parentElement;
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        
        // Wrap the pre element
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = 'Copy';
        copyButton.setAttribute('aria-label', 'Copy code to clipboard');
        
        // Add click event
        copyButton.addEventListener('click', function() {
            const text = codeBlock.textContent;
            
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function() {
                    copyButton.textContent = 'Copied!';
                    copyButton.classList.add('copied');
                    
                    setTimeout(function() {
                        copyButton.textContent = 'Copy';
                        copyButton.classList.remove('copied');
                    }, 2000);
                });
            } else {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                copyButton.textContent = 'Copied!';
                copyButton.classList.add('copied');
                
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                    copyButton.classList.remove('copied');
                }, 2000);
            }
        });
        
        wrapper.appendChild(copyButton);
    });
}

// Generate table of contents
function generateTableOfContents() {
    const tocContainer = document.getElementById('toc');
    if (!tocContainer) return;
    
    const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');
    if (headings.length === 0) {
        tocContainer.parentElement.style.display = 'none';
        return;
    }
    
    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';
    
    headings.forEach(function(heading, index) {
        // Create anchor ID if it doesn't exist
        if (!heading.id) {
            heading.id = 'heading-' + index;
        }
        
        const listItem = document.createElement('li');
        listItem.className = 'toc-item toc-' + heading.tagName.toLowerCase();
        
        const link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        link.className = 'toc-link';
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });
    
    tocContainer.appendChild(tocList);
}

// Calculate and display reading time
function calculateReadingTime() {
    const content = document.querySelector('.post-content');
    if (!content) return;
    
    const wordsPerMinute = 200;
    const text = content.textContent || content.innerText;
    const wordCount = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / wordsPerMinute);
    
    // Update reading time displays
    const readingTimeElements = document.querySelectorAll('.reading-time');
    readingTimeElements.forEach(function(element) {
        if (element.textContent.includes('reading time')) {
            element.textContent = readingTime + ' min read';
        }
    });
}

// Smooth scrolling for anchor links
function initializeSmoothScrolling() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed header
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Back to top functionality
function addBackToTop() {
    const backToTopButton = document.createElement('button');
    backToTopButton.className = 'back-to-top';
    backToTopButton.innerHTML = '↑';
    backToTopButton.setAttribute('aria-label', 'Back to top');
    backToTopButton.style.display = 'none';
    
    document.body.appendChild(backToTopButton);
    
    // Show/hide button based on scroll position
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            backToTopButton.style.display = 'block';
        } else {
            backToTopButton.style.display = 'none';
        }
    });
    
    // Scroll to top on click
    backToTopButton.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Initialize back to top button
addBackToTop();

// Handle MathJax rendering
if (window.MathJax) {
    window.MathJax.startup.promise.then(function() {
        console.log('MathJax is ready');
    });
}

// Add loading indicator for images
function addImageLoadingIndicators() {
    const images = document.querySelectorAll('img');
    
    images.forEach(function(img) {
        img.addEventListener('load', function() {
            img.classList.add('loaded');
        });
        
        img.addEventListener('error', function() {
            img.classList.add('error');
        });
    });
}

addImageLoadingIndicators();