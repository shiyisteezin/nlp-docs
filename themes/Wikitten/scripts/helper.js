hexo.extend.helper.register('toggle_timeline', function() {
  return `
    <div class="widget-wrap">
      <h3 class="widget-title">
        <span id="toggle-archive-widget">${this.__('widget.archives')}</span>
      </h3>
      <div id="archive-widget" style="display: block;">
          <aside id="sidebar-right">
              <%- partial('common/archive', { page: page }) %>
          </aside>
      </div>
    </div>
    <script>
      function toggleVisibility(elementId) {
        var element = document.getElementById(elementId);
        if (element.style.display === 'none' || element.style.display === '') {
          element.style.display = 'block';
        } else {
          element.style.display = 'none';
        }
      }

      document.addEventListener('DOMContentLoaded', function() {
        var toggleButton = document.getElementById('toggle-archive-widget');
        toggleButton.addEventListener('click', function() {
          toggleVisibility('archive-widget');
        });
      });
    </script>
  `;
});
