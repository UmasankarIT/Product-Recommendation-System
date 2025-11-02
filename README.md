🛍️ Product Recommendation System

🎯 Objective

The Product Recommendation System is designed to suggest similar or related products to users based on their search query. It uses content-based filtering powered by TF-IDF and cosine similarity, delivering relevant recommendations in real-time.

🚀 Key Features

✅ Search any product by name
✅ Get instant product recommendations
✅ Clean, modern dark UI
✅ Built with Python + Streamlit
✅ Uses Machine Learning (Content-Based Filtering)
✅ Lightweight, fast, and easy to deploy

🧠 How It Works

The system loads a dataset containing product details such as:

Product Name

Category

Brand

Description

It then processes the data using TF-IDF vectorization to convert text into numerical features.

Cosine Similarity is applied to find the most similar products.

When a user searches for a product, the app displays top matching products based on similarity scores.

🧩 Tech Stack
   Component	Technology Used
   Frontend	Streamlit
   Backend	Python
   Machine Learning	scikit-learn (TF-IDF, Cosine Similarity)
   Data Handling	pandas
   Deployment	Streamlit / Localhost

🖥️ UI Overview

   🎨 Sleek dark theme

   🔍 Search bar for product queries

   📦 List of top recommended products with details

   ⚡ Instant response

🌟 Key Features

✅ Smart search & real-time product suggestions
✅ Clean, dark & modern UI built in Streamlit
✅ Lightweight and fast — ideal for e-commerce demos
✅ Machine Learning-based recommendations using TF-IDF + Cosine Similarity
✅ Extendable with collaborative filtering


📁 Project Structure
Product-Recommendation-System/
│
├── app.py                  # Main Streamlit app
├── products.csv            # Product dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

🧠 How It Works – Process Flow

The system follows a content-based filtering approach using NLP and vector similarity.

🔹 Step-by-Step Process:

   1️⃣ Data Loading

     The app reads product data from products.csv, which includes columns like
     product_name, category, brand, and description.

   2️⃣ Text Preprocessing

     Converts all product-related text into lowercase and removes nulls.

     Combines all text into a single feature column called combined_text.

   3️⃣ TF-IDF Vectorization

     Uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert product text into numerical vectors.

     This helps the model understand the importance of each word.

   4️⃣ Cosine Similarity Calculation

     The app computes Cosine Similarity between product vectors.

     This measures how similar one product is to another based on text features.

   5️⃣ Search & Recommendation

     When a user searches for a product (e.g., “iPhone 15”),
     the app finds the closest match and displays the top recommended products with the highest similarity scores.
 
   6️⃣ Result Display

     The results are displayed in elegant dark-themed cards with:

     🏷️ Product Name

     📂 Category

     💰 Price

     ⭐ Similarity Score

1️⃣ Run the app

streamlit run app.py

📊 Dataset Info

Contains details about multiple categories like:

  📱 Smartphones

  👕 Clothes

  👖 Jeans & Pants

  👟 Footwear

  🕶️ Accessories

  🎧 Electronics

Each product has:

   Name

   Category

   Brand

   Description

💡 Future Enhancements

  🚀 Add collaborative filtering
  🧩 Include image-based similarity
  📈 Integrate user login & feedback
  📱 Build mobile version using Flutter

👨‍💻 Developed By

Umashankar G

🔗 Data Science & AI Enthusiast
📬 Passionate about building intelligent systems with ML & Python
📧 umashankargudivada@gmail.com
