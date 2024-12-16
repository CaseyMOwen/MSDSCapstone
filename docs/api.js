

document.getElementById('postButton').addEventListener('click', () => {
    featdict = getFeatureDict()
    if (featdict != false) {
        document.getElementById('baseline-wrapper').innerHTML = "<br/>Loading... Approximate wait time 30s.<br/>"
        getSavings(featdict)
        .then(data => {
            document.getElementById('baseline-wrapper').innerHTML = ""
            document.getElementById("chart-description").style.display = 'block';

            console.log(data); // JSON data parsed by `response.json()` call
            // const baseline_wrapper = document.getElementById('measures-wrapper')
            // const measures_wrapper = document.getElementById('measures-wrapper')
            
            // Box Plots:
            // var elec_baseline_data = []
            // var fuel_baseline_data = []
            // for (const year_range in data['baseline']) {
            //     var elec_trace = {
            //         y: data['baseline'][year_range]['electricity'],
            //         type: 'box',
            //         name: year_range,
            //         boxpoints: 'suspectedoutliers'
            //     }
            //     var fuel_trace = {
            //         y: data['baseline'][year_range]['other_fuel'],
            //         type: 'box',
            //         name: year_range,
            //         boxpoints: 'suspectedoutliers'
            //     }
            //     elec_baseline_data.push(elec_trace)
            //     fuel_baseline_data.push(fuel_trace)
            // }
            // var elec_baseline_layout = {
            //     title: "Electricity Baseline Usage"
            // }
            // var fuel_baseline_layout = {
            //     title: "Fuel Baseline Usage"
            // }
            // elec_baseline_plot = document.getElementById('elec-baseline-plot')
            // fuel_baseline_plot = document.getElementById('fuel-baseline-plot')
            // Plotly.newPlot(elec_baseline_plot, elec_baseline_data, elec_baseline_layout)
            // Plotly.newPlot(fuel_baseline_plot, fuel_baseline_data, fuel_baseline_layout)

            var elec_baseline_y = []
            var fuel_baseline_y = []
            var elec_baseline_error = []
            var fuel_baseline_error = []
            var elec_baseline_errorminus = []
            var fuel_baseline_errorminus = []
            var baseline_x = []
            console.log()
            for (const year_range in data['baseline']) {
                baseline_x.push(year_range)
                baseline_median_elec = quantile(data['baseline'][year_range]['electricity'], .50)
                baseline_median_fuel = quantile(data['baseline'][year_range]['other_fuel'], .50)
                // The bars are the medians
                elec_baseline_y.push(baseline_median_elec)
                fuel_baseline_y.push(baseline_median_fuel)

                elec_baseline_error.push(quantile(data['baseline'][year_range]['electricity'], .75) - baseline_median_elec)
                fuel_baseline_error.push(quantile(data['baseline'][year_range]['other_fuel'], .75) - baseline_median_fuel)

                elec_baseline_errorminus.push(baseline_median_elec - quantile(data['baseline'][year_range]['electricity'], .25))
                fuel_baseline_errorminus.push(baseline_median_fuel - quantile(data['baseline'][year_range]['other_fuel'], .25))
            }
            var elec_trace = {
                x: baseline_x,
                y: elec_baseline_y,
                // name: year_range,
                error_y: {
                    type: 'data',
                    symmetric: false,
                    array: elec_baseline_error,
                    arrayminus: elec_baseline_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'scatter'
                // boxpoints: 'suspectedoutliers'
            }
            var fuel_trace = {
                x: baseline_x,
                y: fuel_baseline_y,
                // name: year_range,
                error_y: {
                    type: 'data',
                    symmetric: false,
                    array: fuel_baseline_error,
                    arrayminus: fuel_baseline_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'scatter'
                // boxpoints: 'suspectedoutliers'
            }
            var elec_baseline_layout = {
                title: "Electricity Baseline Usage",
                yaxis: {
                    title: 'Annual Energy Baseline Use (kWh)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 5000
                    }
                }
            }
            var fuel_baseline_layout = {
                title: "Fuel Baseline Usage",
                yaxis: {
                    title: 'Annual Energy Baseline Use (therms)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 500
                    }
                }
            }
            elec_baseline_plot = document.getElementById('elec-baseline-plot')
            fuel_baseline_plot = document.getElementById('fuel-baseline-plot')
            Plotly.newPlot(elec_baseline_plot, [elec_trace], elec_baseline_layout)
            Plotly.newPlot(fuel_baseline_plot, [fuel_trace], fuel_baseline_layout)


            // Measures bar plots
            var elec_measure_x = []
            var fuel_measure_x = []
            var elec_measure_error = []
            var fuel_measure_error = []
            var elec_measure_errorminus = []
            var fuel_measure_errorminus = []
            var measure_y = []
            for (const measure_id in data['measures']) {
                measure_median_elec = quantile(data['measures'][measure_id]['electricity'], .50)
                measure_median_fuel = quantile(data['measures'][measure_id]['other_fuel'], .50)
                measure_y.push(data['measures'][measure_id]['name'])
                // The bars are the medians
                elec_measure_x.push(measure_median_elec)
                fuel_measure_x.push(measure_median_fuel)
                
                elec_measure_error.push(quantile(data['measures'][measure_id]['electricity'], .75) - measure_median_elec)
                fuel_measure_error.push(quantile(data['measures'][measure_id]['other_fuel'], .75) - measure_median_fuel)

                elec_measure_errorminus.push(measure_median_elec - quantile(data['measures'][measure_id]['electricity'], .25))
                fuel_measure_errorminus.push(measure_median_fuel - quantile(data['measures'][measure_id]['other_fuel'], .25))
            }
            var elec_trace = {
                x: elec_measure_x,
                y: measure_y,
                name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: elec_measure_error,
                    arrayminus: elec_measure_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var fuel_trace = {
                x: fuel_measure_x,
                y: measure_y,
                // name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: fuel_measure_error,
                    arrayminus: fuel_measure_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var elec_measure_layout = {
                title: "Annual Electricity Baseline Energy Use",
                xaxis: {
                    title: 'Energy Savings, Median and Middle 50% (kWh)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 5000
                    }
                },
                yaxis: {
                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            var fuel_measure_layout = {
                title: "Annual Fuel Baseline Usage",
                xaxis: {
                    title: 'Energy Savings, Median and Middle 50% (therms)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 500
                    }
                },
                yaxis: {

                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            elec_measure_plot = document.getElementById('elec-measure-plot')
            fuel_measure_plot = document.getElementById('fuel-measure-plot')
            Plotly.newPlot(elec_measure_plot, [elec_trace], elec_measure_layout)
            Plotly.newPlot(fuel_measure_plot, [fuel_trace], fuel_measure_layout)

                        // Measures bar plots
            var elec_measure_x = []
            var fuel_measure_x = []
            var elec_measure_error = []
            var fuel_measure_error = []
            var elec_measure_errorminus = []
            var fuel_measure_errorminus = []
            var measure_y = []
            for (const measure_id in data['measures']) {
                measure_median_elec = quantile(data['measures'][measure_id]['electricity'], .50)
                measure_median_fuel = quantile(data['measures'][measure_id]['other_fuel'], .50)
                measure_y.push(data['measures'][measure_id]['name'])
                // The bars are the medians
                elec_measure_x.push(measure_median_elec)
                fuel_measure_x.push(measure_median_fuel)
                
                elec_measure_error.push(quantile(data['measures'][measure_id]['electricity'], .75) - measure_median_elec)
                fuel_measure_error.push(quantile(data['measures'][measure_id]['other_fuel'], .75) - measure_median_fuel)

                elec_measure_errorminus.push(measure_median_elec - quantile(data['measures'][measure_id]['electricity'], .25))
                fuel_measure_errorminus.push(measure_median_fuel - quantile(data['measures'][measure_id]['other_fuel'], .25))
            }
            var elec_trace = {
                x: elec_measure_x,
                y: measure_y,
                name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: elec_measure_error,
                    arrayminus: elec_measure_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var fuel_trace = {
                x: fuel_measure_x,
                y: measure_y,
                // name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: fuel_measure_error,
                    arrayminus: fuel_measure_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var elec_measure_layout = {
                title: "Electricity Measure Savings",
                xaxis: {
                    title: 'Annual Energy Savings, Median and Middle 50% (kWh)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 5000
                    }
                },
                yaxis: {
                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            var fuel_measure_layout = {
                title: "Fuel Measure Savings",
                xaxis: {
                    title: 'Annual Energy Savings, Median and Middle 50% (therms)',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        include: 500
                    }
                },
                yaxis: {

                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            elec_measure_plot = document.getElementById('elec-measure-plot')
            fuel_measure_plot = document.getElementById('fuel-measure-plot')
            Plotly.newPlot(elec_measure_plot, [elec_trace], elec_measure_layout)
            Plotly.newPlot(fuel_measure_plot, [fuel_trace], fuel_measure_layout)

            // Costs bar plots
            var measure_savings_x = []
            var measure_savings_error = []
            var measure_savings_errorminus = []
            var measure_savings_y = []
            for (const measure_id in data['measures']) {
                measure_version = data['measures'][measure_id]['code'].substring(0,6)
                // Multiply measure energy use at each sample * fuel/elec savings at each sample
                measure_elec_savings_array = data['measures'][measure_id]['electricity'].map((num, index) => num * data['cost']['electricity'][measure_version][index])
                measure_fuel_savings_array = data['measures'][measure_id]['other_fuel'].map((num, index) => num * data['cost']['other_fuel'][measure_version][index])

                // measure_fuel_savings_array = data['measures'][measure_id]['other_fuel'].map(function(x) {return x*data['cost']['other_fuel']})
                // ['electricity'].map(function(x) {return x*data['cost']['electricity']})

                // Sum two arrays into one
                measure_savings_array = measure_elec_savings_array.map(function (num, idx) {
                    return num + measure_fuel_savings_array[idx];
                  });
                measure_savings_median = quantile(measure_savings_array, .50)
                measure_savings_y.push(data['measures'][measure_id]['name'])
                // The bars are the medians
                measure_savings_x.push(measure_savings_median)
                
                measure_savings_error.push(quantile(measure_savings_array, .75) - measure_savings_median)

                measure_savings_errorminus.push(measure_savings_median - quantile(measure_savings_array, .25))
            }
            var trace = {
                x: measure_savings_x,
                y: measure_savings_y,
                // name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: measure_savings_error,
                    arrayminus: measure_savings_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var measure_savings_layout = {
                title: "Measure Cost Savings",
                xaxis: {
                    title: 'Annual $ Savings, Median and Middle 50%',
                    rangemode: 'tozero',
                    autorange: true
                },
                yaxis: {
                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            measure_savings_plot = document.getElementById('measure-savings-plot')
            Plotly.newPlot(measure_savings_plot, [trace], measure_savings_layout)

            // Payback bar plots
            var measure_payback_x = []
            var measure_payback_error = []
            var measure_payback_errorminus = []
            var measure_payback_y = []
            for (const measure_id in data['measures']) {
                measure_code = data['measures'][measure_id]['code']
                measure_version = data['measures'][measure_id]['code'].substring(0,6)
                measure_cost = document.getElementById(measure_code + "-cost").value
                // Multiply measure energy use at each sample * fuel/elec savings at each sample
                measure_elec_savings_array = data['measures'][measure_id]['electricity'].map((num, index) => num * data['cost']['electricity'][measure_version][index])
                measure_fuel_savings_array = data['measures'][measure_id]['other_fuel'].map((num, index) => num * data['cost']['other_fuel'][measure_version][index])

                // measure_fuel_savings_array = data['measures'][measure_id]['other_fuel'].map(function(x) {return x*data['cost']['other_fuel']})
                // ['electricity'].map(function(x) {return x*data['cost']['electricity']})

                // Sum two arrays into one
                measure_savings_array = measure_elec_savings_array.map(function (num, idx) {
                    return num + measure_fuel_savings_array[idx];
                  });
                measure_savings_median = quantile(measure_savings_array, .50)
                // The bars are the medians
                measure_savings_x = measure_savings_median
                
                measure_savings_error = quantile(measure_savings_array, .75) - measure_savings_median
                
                measure_savings_errorminus = measure_savings_median - quantile(measure_savings_array, .25)
                if (measure_savings_median > 0) {
                    measure_payback_y.push(data['measures'][measure_id]['name'])
                    measure_payback_x.push(measure_cost/measure_savings_median)
                    measure_payback_error.push(measure_cost/measure_savings_error)
                    measure_payback_errorminus.push(measure_cost/measure_savings_errorminus)
                }
                
            }
            var trace = {
                x: measure_payback_x,
                y: measure_payback_y,
                // name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: measure_payback_error,
                    arrayminus: measure_payback_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var measure_payback_layout = {
                title: "Measure Payback",
                xaxis: {
                    title: 'Annual Payback in Years, Median and Middle 50%',
                    rangemode: 'tozero',
                    autorange: true,
                    autorangeoptions: {
                        clipmax: 30,
                        clipmin: 0
                    }
                },
                yaxis: {
                    showticklabels: true,
                    // tickangle: 45,
                    automargin: true
                }
            }
            measure_payback_plot = document.getElementById('measure-payback-plot')
            Plotly.newPlot(measure_payback_plot, [trace], measure_payback_layout)

        })
        .catch(error => {
            document.getElementById('baseline-wrapper').innerHTML = "<br/>There was an error fetching the data.<br/>"
            console.error('There was a problem with the fetch operation:', error);
        });
    }
});

  function getSavings(featdict={}) {
        
    return fetch("https://msdscapstone-33o5dpumiq-uc.a.run.app/predict", {
    // return fetch("http://localhost:9090/predict", {
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