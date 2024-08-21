document.getElementById('postButton').addEventListener('click', () => {
    featdict = getFeatureDict()
    if (featdict != false) {
        document.getElementById('25th Percentile').innerHTML = "Loading..."
        getSavings(featdict)
        .then(data => {
            console.log(data); // JSON data parsed by `response.json()` call
            document.getElementById('25th Percentile').innerHTML = "First Quartile: " + quantile(data, .25).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
            document.getElementById('Median').innerHTML = "Median: " + quantile(data, .50).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
            document.getElementById('75th Percentile').innerHTML = "75th Percentile: " + quantile(data, .75).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
    }
});

  function getSavings(featdict={}) {
        
    // return fetch("https://msdscapstone-33o5dpumiq-uc.a.run.app/predict", {
    return fetch("http://localhost:9090/predict", {
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
const asc = arr => arr.sort((a, b) => a - b);

const sum = arr => arr.reduce((a, b) => a + b, 0);

const mean = arr => sum(arr) / arr.length;

const quantile = (arr, q) => {
    const sorted = asc(arr);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
        return sorted[base];
    }
};

const std = (arr) => {
    const mu = mean(arr);
    const diffArr = arr.map(a => (a - mu) ** 2);
    return Math.sqrt(sum(diffArr) / (arr.length - 1));
};