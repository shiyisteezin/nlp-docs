(function($) {
    $(document).ready(function() {
        // Load clicked links from localStorage
        var clickedLinks = JSON.parse(localStorage.getItem('clickedLinks')) || [];

        // Ensure all dots are visible on page load
        $('.timeline-icon').css('display', '');

        // Hide dots based on stored clicked links
        clickedLinks.forEach(function(link) {
            $('a[href="' + link + '"]').closest('.timeline-row').find('.timeline-icon').css('display', 'none');
        });

        // Handle click event on timeline-post-title links
        $(document).on('click', '.timeline-post-title a', function(event) {
            var $this = $(this);
            var timelineRow = $this.closest('.timeline-row');
            var timelineIcon = timelineRow.find('.timeline-icon');

            if (timelineIcon.length) {
                // Hide the corresponding dot
                timelineIcon.css('display', 'none');

                // Store the clicked link in localStorage
                var linkHref = $this.attr('href');
                if (!clickedLinks.includes(linkHref)) {
                    clickedLinks.push(linkHref);
                    localStorage.setItem('clickedLinks', JSON.stringify(clickedLinks));
                }
            }
        });

        // Clear localStorage and reset dots when all posts are clicked
        if ($('.timeline-post-title a').length === clickedLinks.length) {
            $(window).on('beforeunload', function() {
                $('.timeline-icon').css('display', '');
                localStorage.removeItem('clickedLinks');
            });
        }
    });

})(jQuery);
