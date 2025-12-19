// DOM Elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const randomBtn = document.getElementById('randomBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');

// Form inputs
const temperatureInput = document.getElementById('temperature');
const humidityInput = document.getElementById('humidity');
const pressureInput = document.getElementById('pressure');
const windSpeedInput = document.getElementById('wind_speed');

// Result elements
const predictionResult = document.getElementById('predictionResult');
const predictionIcon = document.getElementById('predictionIcon');
const predictionText = document.getElementById('predictionText');
const confidenceValue = document.getElementById('confidenceValue');
const rainProbBar = document.getElementById('rainProbBar');
const noRainProbBar = document.getElementById('noRainProbBar');
const rainProb = document.getElementById('rainProb');
const noRainProb = document.getElementById('noRainProb');
const mainInsight = document.getElementById('mainInsight');
const factorsList = document.getElementById('factorsList');
const summaryGrid = document.getElementById('summaryGrid');

// Event Listeners
form.addEventListener('submit', handlePrediction);
randomBtn.addEventListener('click', fillRandomValues);

// Handle form submission
async function handlePrediction(e) {
    e.preventDefault();
    
    // Hide previous results and errors
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingIndicator.style.display = 'block';
    
    // Get form values
    const weatherData = {
        temperature: parseFloat(temperatureInput.value),
        humidity: parseFloat(humidityInput.value),
        pressure: parseFloat(pressureInput.value),
        wind_speed: parseFloat(windSpeedInput.value)
    };
    
    try {
        // Make API call
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(weatherData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

// Display prediction results
function displayResults(data) {
    // Update prediction main result
    const isRain = data.prediction === 'Rain Expected';
    
    predictionResult.className = 'prediction-result ' + (isRain ? 'rain' : 'no-rain');
    predictionIcon.textContent = isRain ? '🌧️' : '☀️';
    predictionText.textContent = data.prediction;
    confidenceValue.textContent = data.confidence.toFixed(1) + '%';
    
    // Update probability bars
    const rainPercentage = data.rain_probability.toFixed(1);
    const noRainPercentage = data.no_rain_probability.toFixed(1);
    
    rainProb.textContent = rainPercentage + '%';
    noRainProb.textContent = noRainPercentage + '%';
    
    rainProbBar.style.width = rainPercentage + '%';
    noRainProbBar.style.width = noRainPercentage + '%';
    
    // Update explanation
    mainInsight.textContent = data.explanation.main_insight;
    
    // Update factors list
    factorsList.innerHTML = '';
    data.explanation.factors.forEach(factor => {
        const factorItem = createFactorElement(factor);
        factorsList.appendChild(factorItem);
    });
    
    // Update input summary
    summaryGrid.innerHTML = `
        <div class="summary-item">
            <div class="summary-label">🌡️ Temperature</div>
            <div class="summary-value">${data.raw_data.temperature}°C</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">💧 Humidity</div>
            <div class="summary-value">${data.raw_data.humidity}%</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">🔽 Pressure</div>
            <div class="summary-value">${data.raw_data.pressure} hPa</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">💨 Wind Speed</div>
            <div class="summary-value">${data.raw_data.wind_speed} km/h</div>
        </div>
    `;
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create factor element
function createFactorElement(factor) {
    const div = document.createElement('div');
    div.className = 'factor-item';
    
    // Determine icon based on feature
    const icons = {
        'temperature': '🌡️',
        'humidity': '💧',
        'pressure': '🔽',
        'wind_speed': '💨',
        'Wind Speed': '💨'
    };
    
    const featureName = factor.feature || factor.name || 'Unknown';
    const icon = icons[featureName] || '📊';
    
    // Format contribution
    const contribution = Math.abs(factor.contribution || 0);
    const direction = factor.direction || 'affects';
    
    div.innerHTML = `
        <div class="factor-icon">${icon}</div>
        <div class="factor-content">
            <div class="factor-text">${factor.description || formatFactorDescription(factor)}</div>
            <div class="factor-contribution">
                Contribution: ${contribution.toFixed(3)} (${direction})
            </div>
        </div>
    `;
    
    return div;
}

// Format factor description
function formatFactorDescription(factor) {
    const featureName = factor.feature || factor.name || 'Unknown feature';
    const value = factor.value !== undefined ? factor.value : 'N/A';
    const direction = factor.direction || 'affects';
    
    return `${capitalize(featureName)}: ${value} - ${direction} the prediction`;
}

// Capitalize first letter
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).replace('_', ' ');
}

// Fill random values
function fillRandomValues() {
    temperatureInput.value = (Math.random() * 40 - 10).toFixed(1);
    humidityInput.value = (Math.random() * 100).toFixed(1);
    pressureInput.value = (Math.random() * 100 + 950).toFixed(1);
    windSpeedInput.value = (Math.random() * 50).toFixed(1);
    
    // Add a subtle animation
    [temperatureInput, humidityInput, pressureInput, windSpeedInput].forEach(input => {
        input.style.transform = 'scale(1.05)';
        setTimeout(() => {
            input.style.transform = 'scale(1)';
        }, 200);
    });
}

// Show error message
function showError(message) {
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Add smooth transition for inputs
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.style.transition = 'all 0.3s ease';
});

// Check API health on page load
async function checkAPIHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status !== 'healthy' || !data.model_loaded) {
            console.warn('API may not be fully ready');
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
}

// Initialize
checkAPIHealth();
