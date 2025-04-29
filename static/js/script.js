// static/js/script.js
$(document).ready(function(){
  $('#search-form').submit(function(event){
    event.preventDefault();
    let title = $('#song-input').val().trim();
    if (!title) return;
    // Clear previous results
    $('#results').empty().append('<p>Loading recommendations...</p>');
    // Query the /recommend endpoint
    $.getJSON('/recommend', { title: title }, function(data){
      $('#results').empty();
      if (data.recommendations.length === 0) {
        $('#results').append('<p>No recommendations found.</p>');
        return;
      }
      // Display each recommendation as a Materialize card
      data.recommendations.forEach(function(rec){
        let card = `
          <div class="card">
            <div class="card-content">
              <span class="card-title">${rec.title}</span>
              <p>${rec.artist}</p>
            </div>
          </div>`;
        $('#results').append(card);
      });
    });
  });
});
