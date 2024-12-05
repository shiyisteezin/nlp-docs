(function($) {
    $(document).ready(function() {
        // Function to seek to a specific time in the YouTube video
        function seekTo(seconds) {
            var $iframe = $('#iframe-yt-video');
            $iframe[0].contentWindow.postMessage(JSON.stringify({
                event: 'command',
                func: 'seekTo',
                args: [seconds, true]
            }), '*');
        }

        // Event listeners for the timestamp buttons
        $('.btn-timestamp').on('click', function(event) {
            event.preventDefault();
            var seconds = $(this).data('seconds');
            seekTo(seconds);
        });

    });
})(jQuery);
