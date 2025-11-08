const form = document.getElementById('fraudForm');
const resultDiv = document.getElementById('result');
const resultContent = document.getElementById('resultContent');
const loading = document.getElementById('loading');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading, hide results
    loading.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    // Collect form data
    const formData = {
        "Transaction Amount": parseFloat(document.getElementById('amount').value),
        "Transaction Hour": parseInt(document.getElementById('hour').value),
        "Account Age Days": parseInt(document.getElementById('accountAge').value),
        "Customer Age": parseInt(document.getElementById('customerAge').value),
        "Payment Method": document.getElementById('paymentMethod').value,
        "Device Used": document.getElementById('device').value,
        "Product Category": document.getElementById('category').value,
        "Shipping Address": document.getElementById('shippingAddress').value,
        "Billing Address": document.getElementById('billingAddress').value
    };
    
    try {
        // Call Flask API
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        // Hide loading
        loading.classList.add('hidden');
        
        // Display result
        displayResult(data);
        
    } catch (error) {
        loading.classList.add('hidden');
        resultContent.innerHTML = `
            <div class="fraud-alert fraud-yes">
                ❌ Error: Could not connect to API. Make sure Flask server is running.
            </div>
        `;
        resultDiv.classList.remove('hidden');
    }
});

function displayResult(data) {
    const isFraud = data.fraud_prediction === 1;
    const probability = (data.probability * 100).toFixed(2);
    
    resultContent.innerHTML = `
        <div class="fraud-alert ${isFraud ? 'fraud-yes' : 'fraud-no'}">
            ${isFraud ? '⚠️ FRAUD DETECTED' : '✅ TRANSACTION SAFE'}
        </div>
        <div class="probability-info">
            <p><strong>Fraud Probability:</strong> ${probability}%</p>
            <p><strong>Decision Threshold:</strong> ${data.threshold * 100}%</p>
            <p><strong>Confidence:</strong> ${isFraud ? probability : (100 - probability)}%</p>
        </div>
    `;
    
    resultDiv.classList.remove('hidden');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}
