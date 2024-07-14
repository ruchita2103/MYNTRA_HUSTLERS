This project is an advanced image similarity checker that leverages machine learning and computer vision techniques to compare images. It is designed to verify the authenticity and similarity of customer-uploaded images against catalog images, making it particularly useful for e-commerce platforms, social media, quality control, and the fashion and apparel industry like MYNTRA . The solution includes background removal, feature extraction using VGG16, color histogram analysis, and Local Binary Patterns (LBP) to provide a comprehensive similarity score. The project is built using Streamlit for an interactive and user-friendly interface.
MYNTRA WEFORSHE HACKATHON
Key Features
Background Removal:

Utilizes the GrabCut algorithm to remove backgrounds from images, ensuring that the similarity comparison focuses solely on the relevant parts of the images.
Feature Extraction:

Extracts deep features using the VGG16 model pretrained on ImageNet.
Analyzes color histograms in the HSV color space.
Computes Local Binary Patterns (LBP) for texture analysis.
Similarity Calculation:

Combines VGG16 features, color histograms, and LBP features to calculate a comprehensive similarity score using cosine similarity.
Adjusts the weights of different features to optimize the similarity measurement.
Interactive Streamlit UI:

Allows users to upload catalog and customer images.
Displays images with and without backgrounds.
Provides detailed and visually appealing feedback based on similarity results.
Hides the raw similarity percentage and instead offers user-friendly messages about the similarity.
Use Cases
E-Commerce:
Verify customer-uploaded images for product reviews, and reward genuine contributions.
Social Media:
Enhance user engagement with content verification, reward programs, and social challenges.
Quality Control:
Ensure product consistency and quality by comparing shipped products with catalog images.
Fashion and Apparel:
Assist customers in finding similar styles and analyze fashion trends.
Benefits
Improved Customer Engagement: Encourages user participation with rewards and verification of genuine content.
Increased Trust: Ensures authenticity, building trust among users.
Enhanced User Experience: Provides personalized recommendations and a sense of community.
Data-Driven Insights: Collects valuable data for trend analysis and decision-making.
Operational Efficiency: Automates image verification, saving time and resources.
