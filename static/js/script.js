document.addEventListener("DOMContentLoaded", function () {
    // Initialize the Chart.js chart
    var ctx = document.getElementById("predictionChart").getContext("2d");
    var predictionChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["Day 1", "Day 2", "Day 3"],
            datasets: [{
                label: "Predicted Price",
                borderColor: "#00ffcc",
                backgroundColor: "rgba(0, 255, 204, 0.2)",
                borderWidth: 2,
                data: []
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { ticks: { color: "#ffffff" } },
                y: { ticks: { color: "#ffffff" } }
            },
            plugins: {
                legend: { labels: { color: "#ffffff" } }
            }
        }
    });

    // Function to fetch stock price prediction
    window.getPrediction = function () {
        let ticker = document.getElementById("ticker").value.trim().toUpperCase();
        let errorMessage = document.getElementById("error-message");
        let loading = document.getElementById("loading");
    
        if (!ticker) {
            errorMessage.innerText = "Please enter a stock ticker!";
            return;
        }
        errorMessage.innerText = "";
        loading.style.display = "block";  // Show loading animation
    
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: ticker })
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = "none";  // Hide loading animation
    
            if (data.error) {
                errorMessage.innerText = data.error;
                return;
            }
    
            // Update the Chart with new predictions
            predictionChart.data.datasets[0].data = data.predicted_prices;
            predictionChart.update();
        })
        .catch(error => {
            loading.style.display = "none";  // Hide loading animation
            errorMessage.innerText = "Error fetching data. Try again.";
            console.error("Fetch error:", error);
        });
    }
});
