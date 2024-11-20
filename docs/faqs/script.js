document.querySelectorAll('.faq-accordion-button').forEach(button => {
    button.addEventListener('click', () => {
        const panel = button.nextElementSibling;
        button.classList.toggle('active');
        if (panel.style.maxHeight) {
            panel.style.maxHeight = null;
        } else {
            panel.style.maxHeight = panel.scrollHeight + 'px';
        }
    });
});
