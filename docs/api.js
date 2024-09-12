

document.getElementById('postButton').addEventListener('click', () => {
    featdict = getFeatureDict()
    if (featdict != false) {
        document.getElementById('baseline-wrapper').innerHTML = "Loading... Approximate wait time 30s."
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
            var measure_cost_x = []
            var measure_cost_error = []
            var measure_cost_errorminus = []
            var measure_cost_y = []
            for (const measure_id in data['measures']) {
                measure_elec_cost_array = data['measures'][measure_id]['electricity'].map(function(x) {return x*data['cost']['electricity']})
                measure_fuel_cost_array = data['measures'][measure_id]['other_fuel'].map(function(x) {return x*data['cost']['other_fuel']})
                // Sum two arrays into one
                measure_cost_array = measure_elec_cost_array.map(function (num, idx) {
                    return num + measure_fuel_cost_array[idx];
                  });
                measure_cost_median = quantile(measure_cost_array, .50)
                measure_cost_y.push(data['measures'][measure_id]['name'])
                // The bars are the medians
                measure_cost_x.push(measure_cost_median)
                
                measure_cost_error.push(quantile(measure_cost_array, .75) - measure_cost_median)

                measure_cost_errorminus.push(measure_cost_median - quantile(measure_cost_array, .25))
            }
            var trace = {
                x: measure_cost_x,
                y: measure_cost_y,
                // name: year_range,
                error_x: {
                    type: 'data',
                    symmetric: false,
                    array: measure_cost_error,
                    arrayminus: measure_cost_errorminus,
                    visible: true,
                    color: 'black'
                },
                type: 'bar',
                orientation: 'h'
                // boxpoints: 'suspectedoutliers'
            }
            var measure_cost_layout = {
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
            measure_cost_plot = document.getElementById('measure-cost-plot')
            Plotly.newPlot(measure_cost_plot, [trace], measure_cost_layout)

// Measure Boxplots:
// var elec_measure_data = []
// var fuel_measure_data = []
            // for (const measure_id in data['measures']) {
            //     var elec_trace = {
            //         x: data['measures'][measure_id]['electricity'],
            //         type: 'box',
            //         name: data['measures'][measure_id]['name'],
            //         boxpoints: 'suspectedoutliers'
            //     }
            //     var fuel_trace = {
            //         x: data['measures'][measure_id]['other_fuel'],
            //         type: 'box',
            //         name: data['measures'][measure_id]['name'],
            //         boxpoints: 'suspectedoutliers'
            //     }
            //     elec_measure_data.push(elec_trace)
            //     fuel_measure_data.push(fuel_trace)
            // }
            // var elec_measure_layout = {
            //     title: "Electricity Savings"
            // }
            // var fuel_measure_layout = {
            //     title: "Fuel Savings"
            // }
            // elec_measure_plot = document.getElementById('elec-measure-plot')
            // fuel_measure_plot = document.getElementById('fuel-measure-plot')
            // Plotly.newPlot(elec_measure_plot, elec_measure_data, elec_measure_layout)
            // Plotly.newPlot(fuel_measure_plot, fuel_measure_data, fuel_measure_layout)


                // measure_results = document.createElement('div')
                // measure_results.setAttribute("class", "measure-results")
                // measure_results.setAttribute("id", measure_id + "-measure-results")
                // if (measure_id == "0") {
                //     baseline_wrapper.appendChild(measure_results)
                // } else {
                //     measures_wrapper.appendChild(measure_results)
                // }

                // measure_name = document.createElement('div')
                // measure_name.setAttribute("class", "measure-name")
                // measure_name.innerHTML = data[measure_id]['name']
                // measure_results.appendChild(measure_name)

                // measure_desc = document.createElement('div')
                // measure_desc.setAttribute("class", "measure-desc")
                // measure_desc.innerHTML = data[measure_id]['description']
                // measure_results.appendChild(measure_desc)

                // elec_wrapper = document.createElement('div')
                // elec_wrapper.setAttribute("class", "elec_wrapper")
                // measure_results.appendChild(elec_wrapper)

                // elec_title = document.createElement('div')
                // elec_title.setAttribute("class", "elec-title")
                // elec_title.innerHTML = 'Median Electric Savings:'
                // elec_wrapper.appendChild(elec_title)

                // elec_value = document.createElement('div')
                // elec_value.setAttribute("class", "elec-value")
                // elec_value.innerHTML = quantile(data[measure_id]['other_fuel'], .50).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
                // elec_wrapper.appendChild(elec_value)

                // fuel_wrapper = document.createElement('div')
                // fuel_wrapper.setAttribute("class", "fuel_wrapper")
                // measure_results.appendChild(fuel_wrapper)

                // fuel_title = document.createElement('div')
                // fuel_title.setAttribute("class", "fuel-title")
                // fuel_title.innerHTML = 'Median Other Fuel Savings:'
                // fuel_wrapper.appendChild(fuel_title)

                // fuel_value = document.createElement('div')
                // fuel_value.setAttribute("class", "fuel-value")
                // fuel_value.innerHTML = quantile(data[measure_id]['electricity'], .50).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
                // fuel_wrapper.appendChild(fuel_value)            
            // }
            // document.getElementById('25th Percentile').innerHTML = "First Quartile: " + quantile(data, .25).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
            // document.getElementById('Median').innerHTML = "Median: " + quantile(data, .50).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
            // document.getElementById('75th Percentile').innerHTML = "75th Percentile: " + quantile(data, .75).toLocaleString("en-US", {maximumFractionDigits:0}) + " kWh"
        })
        .catch(error => {
            document.getElementById('baseline-wrapper').innerHTML = "There was an error fetching the data. Try specifying different inputs - there are still bugs (I know there is a bug when selecting heating system as mini-splits)"
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