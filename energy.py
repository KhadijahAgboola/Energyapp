import streamlit as st
from PIL import Image
import streamlit as st
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Energy Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and set the background image
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        image_bytes = image.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")  # Encode as base64
    bg_style = f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{encoded_image}") no-repeat center fixed;
        background-size: cover;
        opacity: 1.2; /* Adjust for lighter shade */
    }}
    .big-bold-text {{
        font-size: 30px; /* Optional shadow for contrast */
        color: black;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Usage
set_background("energypic.jpg")


# App title and problem statement
# Set the title with larger font size
st.markdown("""
<style>
.big-title {
    font-size: 50px;
    font-weight: bold;
    color: black; /* Adjust color as needed */
}
.problem-statement {
    font-size: 25px;
    line-height: 1.6; /* Adjust for readability */
    color: black;
}
</style>
<div class="big-title">Energy Production & Market Forecasting</div>
""", unsafe_allow_html=True)

# Problem Statement
st.markdown("""
<div class="problem-statement"><h2>Problem Statement</h2>
"We are seeking a skilled AI & Machine Learning specialist to collaborate on developing predictive models for energy production, pricing, and market forecasting. 
The role involves designing and implementing machine learning algorithms to analyze weather data, historical production patterns, market trends, 
and other key variables to optimize decision-making in the renewable energy sector."
</div>
""", unsafe_allow_html=True)

# Add space before the section
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# Solution description
st.markdown("""
<div class="problem-statement"><h2>Solution</h2>
    The solution to this problem will be grouped into three sections:
    <ol>
        <li style="font-size: 25px;"><b>Trend Analysis</b></li>
        <li style="font-size: 25px;"><b>Energy Forecasting using Linear Regression</b></li>
        <li style="font-size: 25px;"><b>Optimization Recommendations</b></li>
    </ol>
    I trained the models using the renewable energy production dataset from Kaggle. <a href="https://www.kaggle.com/datasets/ahmedgaitani/global-renewable-energy/data" target="_blank">Dataset Link</a>
</div>
""", unsafe_allow_html=True)

# Add space before the section
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)


# Section 1: Trend Analysis
st.header("1. Trend Analysis")

# Add the image with updated parameter for width
st.image("trend.PNG", caption="Trends in Renewable Energy Production", use_container_width=True)

# Details below the image with larger font size
st.markdown("""
<style>
    .trend-summary {
        font-size: 50px; /* Adjust the size to your preference */
        line-height: 1.6; /* Increase line spacing for better readability */
        color: black;
    }
    .trend-summary h3 {
        font-size: 30px; /* Subheading size */
        margin-top: 20px;
        margin-bottom: 10px;
        color: black; 
    }
    .trend-summary ul {
        font-size: 40px; /* Match paragraph font size */
        margin-left: 20px; /* Indent for lists */
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .trend-summary ul li {
        margin-bottom: 5px; /* Space between list items */
    }
</style> 

<div class="trend-summary">
<h3>Trend Analysis Summary</h3></div>
<p style="font-size: 25px;">From the trend analysis of renewable energy production data:</p>

<h3>Global Energy Trends:</h3>
<div class="trend-summary ul"><ul>
    <li style="font-size: 25px;">Total renewable energy production experienced significant declines in <strong>2003, 2011, 2016, and 2020</strong>, indicating potential global disruptions or economic factors affecting energy production during these years.</li>
    <li style="font-size: 25px;">Conversely, total energy production peaked in <strong>2008, 2013, 2017, and 2021</strong>, marking these years as periods of increased renewable energy generation.</li>
</ul>

<h3>Country Contributions:</h3>
<ul>
    <li style="font-size: 25px;"><strong>India and France</strong> emerged as the leading producers of renewable energy, each generating approximately <strong>70,000 GWh</strong>, showcasing their strong commitment to renewable energy initiatives.</li>
</ul>

<h3>Distribution Patterns:</h3>
<ul>
    <li style="font-size: 25px;">The most frequent total renewable energy production value globally was around <strong>30,000 GWh</strong>, indicating this as a common production level for many countries.</li>
    <li style="font-size: 25px;">Notably, <strong>outliers</strong> were observed only in the <strong>UK's distribution</strong> of total renewable energy, suggesting unique conditions or anomalies in the country's production patterns. Other countries exhibited consistent distributions with no outliers.</li>
</ul>

<h3>Energy Source Trends:</h3>
<ul>
    <li style="font-size: 25px;"><strong>HydroEnergy</strong> consistently produced the highest energy output over the years, highlighting its dominance as a renewable energy source.</li>
    <li style="font-size: 25px;">This was followed by <strong>WindEnergy</strong>, which also contributed significantly to global energy production, reflecting its growing adoption worldwide.</li>
</ul>

<p style="font-size: 25px;">These insights underscore key global and regional trends, as well as the importance of specific renewable energy sources in driving overall production.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  .summary {
        font-size: 30px; /* Match paragraph font size */
        color=black;
        margin-left: 20px; /* Indent for lists */
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""",
unsafe_allow_html=True
)
# Add space before the section
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)


# Load data (Replace with actual file paths)
train = pd.read_csv("train_dataset.csv")

# Linear Regression using only 'Year' as the feature
X_train = train[['Year']]
y_train = train['TotalRenewableEnergy']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Section 2: Energy Forecasting using Linear Regression
st.header("2. Energy Forecasting using Linear Regression")

# Explanation of the section
st.markdown("""
<div class="summary">
This section uses a Linear Regression model to predict Total Renewable Energy production based on the year of production. 
The model is trained on historical data and evaluated on a test dataset.</div>""",     unsafe_allow_html=True
)

# Add space before the section
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# Prompt user for a future year
st.subheader("Forecast Total Renewable Energy")
future_year = st.number_input("Enter a future year (e.g., 2025):", min_value=2024, step=1)

# Predict Total Renewable Energy for the entered year
if future_year:
    future_prediction = model.predict([[future_year]])[0]
    st.write(f"""<div class="summary">Predicted Total Renewable Energy for the year {future_year} is {future_prediction:.2f} GWh
      </div>""",
            unsafe_allow_html=True
        )

# Add space before the section
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# Inject custom CSS for styling
st.markdown("""
    <style>
    .optimization-recommendations {
        font-size: 20px;
        line-height: 1.8;
    }
    .optimization-recommendations h2 {
        font-size: 28px;
        margin-bottom: 20px;
    }
    .optimization-recommendations ul {
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .optimization-recommendations li {
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


# Section 3: Optimization Recommendations
st.header("3. Optimization Recommendations")

st.markdown("""
<div class="optimization-recommendations">
    <p style="font-size: 30px;">Here are a few practical steps to improve renewable energy production and make the most of existing resources:</p>
    <ul style="font-size: 25px;">
        <li style="font-size: 25px;">
            <strong>Focus on Hydro and Wind Energy:</strong> 
            HydroEnergy consistently produces the most power, so keeping dams and hydro plants in top shape is key. 
            WindEnergy is the second-biggest contributor, so expanding wind farms can give energy production a solid boost.
        </li>
        <li style="font-size: 25px;">
            <strong>Learn from Top Performers:</strong> 
            India and France are leading the pack, producing around 70,000 GWh each. Other countries can learn from their approaches and adopt similar strategies.
        </li>
        <li style="font-size: 25px;">
            <strong>Tackle Unique Challenges:</strong> 
            The UK shows some unusual patterns with energy production outliers. Taking a closer look might uncover issues or opportunities to improve.
        </li>
        <li style="font-size: 25px;">
            <strong>Use Smarter Tech:</strong> 
            Tools like smart grids, AI for better forecasting, and real-time monitoring can help make energy production more efficient.
        </li>
        <li style="font-size: 25px;">
            <strong>Work Together:</strong> 
            High-performing countries can share knowledge and tools with others to lift everyone’s game. Global collaboration could make a big difference.
        </li>
        <li style="font-size: 25px;">
            <strong>Support Clean Energy Policies:</strong> 
            Governments should make it easier and cheaper to invest in renewable energy through tax breaks and incentives. Cutting red tape for new projects will help, too.
        </li>
    </ul>
    <p style="font-size: 25px;">
        These steps can help countries maximize renewable energy potential, cut emissions, and build a cleaner, more sustainable future for everyone.
    </p>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
