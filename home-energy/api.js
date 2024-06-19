document.getElementById('postButton').addEventListener('click', () => {
    document.getElementById('result').innerHTML = "Loading..."
    getSavings(getFeatureDict())
      .then(data => {
        console.log(data); // JSON data parsed by `response.json()` call
        document.getElementById('result').innerHTML = data[0].toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
      })
      .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
      });
  });

  function getSavings(featdict={}) {
        
    return fetch("https://msdscapstone-33o5dpumiq-uc.a.run.app/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(featdict)
    })
    .then(response => {
        if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json(); 
    });
}