let ft_titles = {};

function getRealElement(selector) {
    let elem = gradioApp().getElementById(selector);
    if (elem) {
        let child = elem.querySelector('#' + selector);
        if (child) {
            return child;
        } else {
            return elem;
        }
    }
    return elem;
}

function setHints() {
    // If more_hints is defined, we'll add the entries to the ft_titles object
    if (typeof more_hints !== 'undefined') {
        console.log("Adding more hints!");
        for (let key in more_hints) {
            ft_titles[key] = more_hints[key];
        }
    }
    gradioApp().querySelectorAll('span, button, select, p').forEach(function (span) {
        let tooltip = ft_titles[span.textContent];
        if (span.disabled || span.classList.contains(".\\!cursor-not-allowed")) {
            tooltip = "Select or Create a Model."
        }

        if (!tooltip) {
            tooltip = ft_titles[span.value];
        }

        if (!tooltip) {
            for (const c of span.classList) {
                if (c in ft_titles) {
                    tooltip = ft_titles[c];
                    break;
                }
            }
        }

        if (tooltip) {
            span.title = tooltip;
        }

    });

    gradioApp().querySelectorAll('select').forEach(function (select) {
        if (select.onchange != null) return;
        select.onchange = function () {
            select.title = ft_titles[select.value] || "";
        }
    });

    gradioApp().querySelectorAll('.gallery-item').forEach(function (btn) {
        if (btn.onchange != null) return;
        btn.onchange = function () {
            // Dummy function, so we don't keep setting up the observer.
        }
        checkPrompts();
        const options = {
            attributes: true
        }

        function callback(mutationList, observer) {
            mutationList.forEach(function (mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    checkPrompts();
                }
            });
        }

        const observer = new MutationObserver(callback);
        observer.observe(btn, options);

    });

}

// Do a thing when the UI updates
onUiUpdate(function () {
    setHints();
});

