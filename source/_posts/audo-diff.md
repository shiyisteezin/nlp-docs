---
title: Auto Differentiation
date: 2024-07-14 13:40
categories:
  - Math
tags:
  - math
---

### Intro to Automatic Differentiation

In this blog, we will go through the foundations behind Automatic Differentiation.

<div style="margin-left:1px; margin-top: 30px">
  <iframe id="iframe-yt-video" width="100%" height="520" src="https://www.youtube.com/embed/56WUlMEeAuA?enablejsapi=1&autoplay=1" frameborder="0"></iframe>
</div>

#### The Breakdown of The Video Into Parts
<script src="https://www.youtube.com/iframe_api"></script>

<script>
    var player;

    // This function is called by the YouTube IFrame API when it's ready
    function onYouTubeIframeAPIReady() {
        player = new YT.Player('iframe-yt-video', {
            events: {
                'onReady': onPlayerReady
            }
        });
    }

    // This function is called when the player is ready
    function onPlayerReady(event) {
        console.log("YouTube Player is ready");

        // Attach event listeners to the buttons
        document.querySelectorAll('.btn-timestamp').forEach(function(button) {
            button.addEventListener('click', function() {
                var seconds = parseInt(this.getAttribute('data-seconds'), 10);
                seekTo(seconds);
            });
        });
    }

    // Function to seek to the specified time in the video
    function seekTo(seconds) {
        if (player && typeof player.seekTo === 'function') {
            player.pauseVideo(); // Pause the video first
            player.seekTo(seconds, true); // Seek to the specified time
            setTimeout(function() {
                player.playVideo(); // Attempt to play after a short delay
            }, 500); // Adjust delay as needed
            console.log("Seeking to:", seconds, "seconds");
        } else {
            console.log("Player is not ready yet");
        }
    }
</script>

<div class="video-timestamps">
  <button class="btn-timestamp" data-seconds="0">Introduction</button>
  <button class="btn-timestamp" data-seconds="53">Topic 1 - Diff in ML</button>
  <button class="btn-timestamp" data-seconds="316">Topic 2 - Numerical Diff</button>
  <button class="btn-timestamp" data-seconds="697">Topic 2 - Gradient Checking</button>
  <button class="btn-timestamp" data-seconds="848">Topic 3 - Symbolic Diff</button>
  <button class="btn-timestamp" data-seconds="1108">Topic 4 - Computational Graphs</button>
  <button class="btn-timestamp" data-seconds="1367">Topic 5 - Forward Automatic Diff</button>
  <button class="btn-timestamp" data-seconds="1992">Topic 6 - Reverse Auto Diff</button>
  <button class="btn-timestamp" data-seconds="2532">Topic 6 - Reverse AD Algorithm</button>
  <button class="btn-timestamp" data-seconds="3299">Topic 7 - RAD vs Backprop</button>

</div>


#### Important Transcripts of the Video
