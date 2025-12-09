import streamlit as st
import requests
import base64
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import json
import singlestoredb as s2
import time
import re

# Import SentenceTransformer as fallback for embeddings
SENTENCE_TRANSFORMER_AVAILABLE = False
embedding_model = None

def get_embedding_model():
    """Lazy load SentenceTransformer to avoid blocking Streamlit startup"""
    global embedding_model, SENTENCE_TRANSFORMER_AVAILABLE
    if embedding_model is None and SENTENCE_TRANSFORMER_AVAILABLE == False:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("BAAI/bge-m3")
            SENTENCE_TRANSFORMER_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMER_AVAILABLE = False
        except Exception as e:
            SENTENCE_TRANSFORMER_AVAILABLE = False
    return embedding_model

# Load environment variables from .env file
load_dotenv()

# Model configuration - using lighter model to avoid quota issues
LLM_PRIMARY = "gemini-1.5-flash"  # Lighter model with better free tier
LLM_FALLBACK = "gemini-1.5-flash"  # Same as primary for now

# Get the API key from environment variables
subscription_key = "731bcfac-aef2-4541-88ee-1dc114b017a4"
xai_api_key = "AIzaSyA02JdZFZ3Xjj26ThJhhJQ7anhrbrI66h8"
#xai_api_key = "AIzaSyALeNep6neidGStoM1qax9GAxu1_GFNLIo"
sarvamurl = "https://api.sarvam.ai/text-to-speech"
sarvamheaders = {
    "accept": "application/json",
    "content-type": "application/json",
    "api-subscription-key": subscription_key
}

# Initialize OpenAI client for Gemini
LLMclient=OpenAI(
  api_key=xai_api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
Nclient = OpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)

# Load CSS from file
with open("style.css", "r") as f:
    css = f.read()

# Add background image to CSS
css += """
body {
    background-image: url('https://fillyou.in/wp-content/uploads/2024/01/service_6-scaled-1.webp'); /* Update with your image path */
    background-size: cover; /* Cover the entire page */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Prevent the image from repeating */
    height: 100vh; /* Ensure the body takes full height */
}
"""

# Inject CSS into Streamlit
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
def create_connection():
    return s2.connect('admin:X8MBbWxI1NiuG6RGPhyIQcr7lz4oseOY@svc-8bd4e6d7-dd92-449e-b8af-56828e3aea12-dml.aws-mumbai-1.svc.singlestore.com:3306/miniDB')

# Function to send audio to Sarvam API for speech recognition
def transcribe_audio(audio_file, subscription_key, language_code):
    url = "https://api.sarvam.ai/speech-to-text"
    
    # Prepare the payload with model and language code
    payload = {
        'model': 'saarika:v1',
        'language_code': language_code,
        'with_timesteps': 'false'
    }
    
    # Prepare files for the request
    files = [
        ('file', ('audio.wav', audio_file, 'audio/wav'))
    ]
    
    # Set headers including your API subscription key
    headers = {
        'api-subscription-key': subscription_key
    }
    
    # Make the POST request to the API
    response = requests.post(url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        return response.json().get('transcript', 'No transcript available.')
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to get embeddings and nearest neighbors
def get_embeddings_and_neighbors(sentence):
    embeddings_json = None
    
    # Try NVIDIA API first
    try:
        response = Nclient.embeddings.create(
            input=[sentence],
            model="baai/bge-m3",
            encoding_format="float",
            extra_body={"truncate": "NONE"}
        )
        embeddings_json = json.dumps(response.data[0].embedding)  # Convert to JSON
        st.success("✓ Using NVIDIA API for embeddings")
    except Exception as e:
        # Fallback to SentenceTransformer if NVIDIA API fails
        model = get_embedding_model()
        if model is not None:
            try:
                st.info("NVIDIA API unavailable. Using SentenceTransformer fallback...")
                embeddings = model.encode(sentence)
                embeddings_json = json.dumps(embeddings.tolist())
                st.success("✓ Using SentenceTransformer for embeddings")
            except Exception as e2:
                st.error(f"Both embedding methods failed. NVIDIA: {str(e)}, SentenceTransformer: {str(e2)}")
                # Return mock results
                mock_html = "<html><body><h1>Mock Webpage</h1><p>This is a mock result because the embeddings API is not available.</p></body></html>"
                return [(0, "Mock Idea", mock_html, None, 0.0)]
        else:
            st.warning(f"Embeddings API error: {str(e)}. Using mock data. Please check your NVIDIA API key or install sentence-transformers.")
            # Mock result: id, llm_generated_idea, text, embedding, score
            mock_html = "<html><body><h1>Mock Webpage</h1><p>This is a mock result because the embeddings API is not available.</p></body></html>"
            return [(0, "Mock Idea", mock_html, None, 0.0)]

    # Try to connect to the database and search for nearest neighbors
    try:
        with create_connection() as conn:
            with conn.cursor() as cur:
                # Search for nearest neighbors using the new query format
                cur.execute(""" 
                    SELECT id, llm_generated_idea, text, embedding, embedding <-> %s AS score 
                    FROM webgen 
                    ORDER BY score 
                    LIMIT 5
                """, (embeddings_json,))  # Pass embeddings as a JSON array

                # Fetch the results
                results = cur.fetchall()
                if results:
                    with open("embeddings.txt", "w") as f:
                        for row in results:
                            f.write(str(row) + "\n")  # Write each row as a string to the file
                    st.success(f"✓ Found {len(results)} similar designs in database")
                    return results  # Return the fetched results
                else:
                    st.info("No similar designs found in database. Using default template.")
                    return None
    except Exception as e:
        # If database connection fails, return None to use default processing
        st.warning(f"Database connection failed: {e}. Proceeding without database lookup.")
        return None

# Function to generate HTML using template fallback
def generate_html_template(user_text):
    """Generate a relevant HTML template when LLM APIs fail"""
    # Extract key words and context from user input
    keywords = user_text.lower()

    # Determine website type and content with enhanced theming
    website_type = "Business"
    title = "My Website"
    description = "Welcome to our website"
    features = ["Feature 1", "Feature 2", "Feature 3"]
    color_scheme = "from-blue-500 to-purple-600"
    hero_image = "https://image.pollinations.ai/prompt/modern%20business%20office%20with%20team%20collaboration.png"
    theme_icons = ["💼", "🚀", "💎"]
    theme_content = {
        "hero_title": "Welcome to Our Business",
        "hero_subtitle": "Delivering excellence in every service",
        "about_title": "About Our Company",
        "about_content": "We are dedicated to providing exceptional services and creating memorable experiences for our clients.",
        "services": ["Consulting", "Development", "Support"]
    }

    if "restaurant" in keywords or "food" in keywords or "dining" in keywords or "cafe" in keywords or "menu" in keywords:
        website_type = "Restaurant"
        title = "Gourmet Restaurant"
        description = "Experience culinary excellence with our chef's special menu"
        features = ["Fresh Ingredients", "Expert Chefs", "Cozy Ambiance"]
        color_scheme = "from-orange-500 to-red-600"
        hero_image = "https://image.pollinations.ai/prompt/elegant%20restaurant%20interior%20with%20dining%20tables.png"
        theme_icons = ["🍽️", "👨‍🍳", "🏠"]
        theme_content = {
            "hero_title": "Welcome to Gourmet Dining",
            "hero_subtitle": "Where every meal tells a story",
            "about_title": "Our Culinary Journey",
            "about_content": "From farm to table, we bring you the finest ingredients prepared by master chefs in an ambiance that feels like home.",
            "services": ["Fine Dining", "Private Events", "Catering"]
        }
    elif "hotel" in keywords or "booking" in keywords or "resort" in keywords or "accommodation" in keywords:
        website_type = "Hotel"
        title = "Luxury Hotel & Resort"
        description = "Your perfect getaway destination"
        features = ["Luxury Rooms", "Spa & Wellness", "Fine Dining"]
        color_scheme = "from-teal-500 to-blue-600"
        hero_image = "https://image.pollinations.ai/prompt/luxury%20hotel%20lobby%20with%20ocean%20view.png"
        theme_icons = ["🏨", "🧖", "🍽️"]
        theme_content = {
            "hero_title": "Luxury Awaits",
            "hero_subtitle": "Experience world-class hospitality",
            "about_title": "Your Home Away From Home",
            "about_content": "Indulge in our luxurious accommodations, rejuvenate at our world-class spa, and savor exquisite cuisine.",
            "services": ["Luxury Rooms", "Spa Treatments", "Fine Dining"]
        }
    elif "shop" in keywords or "store" in keywords or "ecommerce" in keywords or "buy" in keywords or "shopping" in keywords:
        website_type = "E-commerce"
        title = "Online Store"
        description = "Shop the latest trends and exclusive deals"
        features = ["Fast Delivery", "Secure Payment", "24/7 Support"]
        color_scheme = "from-green-500 to-emerald-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20online%20shopping%20storefront.png"
        theme_icons = ["🛒", "🚚", "💳"]
        theme_content = {
            "hero_title": "Shop with Confidence",
            "hero_subtitle": "Discover amazing products at unbeatable prices",
            "about_title": "Your Trusted Online Store",
            "about_content": "We offer a curated selection of premium products with fast shipping, secure payments, and exceptional customer service.",
            "services": ["Fast Delivery", "Secure Checkout", "Customer Support"]
        }
    elif "portfolio" in keywords or "resume" in keywords or "personal" in keywords or "freelancer" in keywords:
        website_type = "Portfolio"
        title = "Professional Portfolio"
        description = "Showcasing my work and expertise"
        features = ["Projects", "Skills", "Experience"]
        color_scheme = "from-indigo-500 to-purple-600"
        hero_image = "https://image.pollinations.ai/prompt/creative%20portfolio%20workspace%20with%20laptop.png"
        theme_icons = ["💼", "🎨", "📈"]
        theme_content = {
            "hero_title": "Hello, I'm a Creative Professional",
            "hero_subtitle": "Bringing ideas to life through design and innovation",
            "about_title": "About Me",
            "about_content": "Passionate about creating meaningful experiences through thoughtful design and innovative solutions.",
            "services": ["UI/UX Design", "Web Development", "Branding"]
        }
    elif "blog" in keywords or "news" in keywords or "journal" in keywords or "writing" in keywords:
        website_type = "Blog"
        title = "My Blog"
        description = "Sharing thoughts and insights"
        features = ["Latest Posts", "Categories", "About"]
        color_scheme = "from-pink-500 to-rose-600"
        hero_image = "https://image.pollinations.ai/prompt/cozy%20blogging%20workspace%20with%20books.png"
        theme_icons = ["📝", "📚", "💭"]
        theme_content = {
            "hero_title": "Welcome to My Blog",
            "hero_subtitle": "Exploring ideas, sharing insights, inspiring change",
            "about_title": "About This Blog",
            "about_content": "A space for thoughtful discussions, creative ideas, and meaningful conversations about topics that matter.",
            "services": ["Articles", "Tutorials", "Insights"]
        }
    elif "fitness" in keywords or "gym" in keywords or "workout" in keywords or "health" in keywords:
        website_type = "Fitness"
        title = "Fitness Center"
        description = "Transform your body and mind"
        features = ["Personal Training", "Group Classes", "Nutrition Plans"]
        color_scheme = "from-red-500 to-orange-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20gym%20with%20equipment%20and%20people.png"
        theme_icons = ["💪", "🏃", "🥗"]
        theme_content = {
            "hero_title": "Transform Your Life",
            "hero_subtitle": "Achieve your fitness goals with expert guidance",
            "about_title": "Your Fitness Journey Starts Here",
            "about_content": "Join our community of fitness enthusiasts and transform your body and mind with our comprehensive programs.",
            "services": ["Personal Training", "Group Classes", "Nutrition Coaching"]
        }
    elif "education" in keywords or "school" in keywords or "learning" in keywords or "course" in keywords or "teaching" in keywords:
        website_type = "Education"
        title = "Learning Platform"
        description = "Empowering education for everyone"
        features = ["Expert Teachers", "Interactive Courses", "Certificates"]
        color_scheme = "from-blue-500 to-cyan-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20classroom%20with%20students%20and%20teacher.png"
        theme_icons = ["🎓", "📖", "🏆"]
        theme_content = {
            "hero_title": "Learn. Grow. Succeed.",
            "hero_subtitle": "Unlock your potential with quality education",
            "about_title": "Empowering Learners Worldwide",
            "about_content": "We provide accessible, high-quality education through interactive courses taught by industry experts.",
            "services": ["Online Courses", "Certifications", "Skill Development"]
        }
    elif "healthcare" in keywords or "medical" in keywords or "hospital" in keywords or "clinic" in keywords:
        website_type = "Healthcare"
        title = "Medical Center"
        description = "Caring for your health and wellness"
        features = ["Expert Doctors", "Modern Facilities", "Patient Care"]
        color_scheme = "from-emerald-500 to-teal-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20medical%20facility%20with%20doctors.png"
        theme_icons = ["⚕️", "🏥", "❤️"]
        theme_content = {
            "hero_title": "Your Health, Our Priority",
            "hero_subtitle": "Comprehensive healthcare services with compassion",
            "about_title": "Committed to Your Well-being",
            "about_content": "Our team of experienced healthcare professionals is dedicated to providing exceptional medical care.",
            "services": ["Primary Care", "Specialized Treatment", "Emergency Services"]
        }
    elif "travel" in keywords or "tourism" in keywords or "vacation" in keywords or "trip" in keywords:
        website_type = "Travel"
        title = "Travel Agency"
        description = "Discover amazing destinations worldwide"
        features = ["Custom Tours", "Best Deals", "Expert Guides"]
        color_scheme = "from-cyan-500 to-blue-600"
        hero_image = "https://image.pollinations.ai/prompt/beautiful%20travel%20destination%20with%20mountains.png"
        theme_icons = ["✈️", "🌍", "🏖️"]
        theme_content = {
            "hero_title": "Explore the World",
            "hero_subtitle": "Create unforgettable memories with our travel experiences",
            "about_title": "Your Gateway to Adventure",
            "about_content": "From exotic destinations to cultural experiences, we craft personalized journeys for every traveler.",
            "services": ["Custom Tours", "Adventure Trips", "Cultural Experiences"]
        }
    elif "technology" in keywords or "tech" in keywords or "software" in keywords or "app" in keywords:
        website_type = "Technology"
        title = "Tech Solutions"
        description = "Innovative technology solutions for modern businesses"
        features = ["Custom Software", "Cloud Services", "Tech Support"]
        color_scheme = "from-slate-500 to-gray-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20technology%20workspace%20with%20computers.png"
        theme_icons = ["💻", "☁️", "🔧"]
        theme_content = {
            "hero_title": "Powering Digital Innovation",
            "hero_subtitle": "Transforming businesses through cutting-edge technology",
            "about_title": "Leading Technology Solutions",
            "about_content": "We deliver innovative software solutions and cloud services to help businesses thrive in the digital age.",
            "services": ["Software Development", "Cloud Solutions", "IT Consulting"]
        }
    elif "music" in keywords or "band" in keywords or "artist" in keywords or "concert" in keywords:
        website_type = "Music"
        title = "Music Studio"
        description = "Where creativity meets melody"
        features = ["Recording Studio", "Live Performances", "Music Lessons"]
        color_scheme = "from-purple-500 to-pink-600"
        hero_image = "https://image.pollinations.ai/prompt/music%20studio%20with%20instruments%20and%20equipment.png"
        theme_icons = ["🎵", "🎤", "🎸"]
        theme_content = {
            "hero_title": "Feel the Rhythm",
            "hero_subtitle": "Creating music that moves your soul",
            "about_title": "Your Musical Journey",
            "about_content": "From recording sessions to live performances, we help artists bring their musical visions to life.",
            "services": ["Recording Services", "Live Events", "Music Production"]
        }
    elif "art" in keywords or "gallery" in keywords or "painting" in keywords or "creative" in keywords:
        website_type = "Art"
        title = "Art Gallery"
        description = "Discover and collect extraordinary art"
        features = ["Contemporary Art", "Artist Showcases", "Art Consultations"]
        color_scheme = "from-amber-500 to-orange-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20art%20gallery%20with%20paintings.png"
        theme_icons = ["🎨", "🖼️", "✨"]
        theme_content = {
            "hero_title": "Where Art Comes Alive",
            "hero_subtitle": "Discover extraordinary works by talented artists",
            "about_title": "Celebrating Artistic Expression",
            "about_content": "We showcase contemporary art from emerging and established artists, creating spaces for creative dialogue.",
            "services": ["Art Exhibitions", "Artist Representation", "Art Consulting"]
        }
    elif "sports" in keywords or "athletic" in keywords or "game" in keywords or "competition" in keywords:
        website_type = "Sports"
        title = "Sports Academy"
        description = "Excellence in athletic performance and training"
        features = ["Professional Coaching", "State-of-the-art Facilities", "Youth Programs"]
        color_scheme = "from-lime-500 to-green-600"
        hero_image = "https://image.pollinations.ai/prompt/modern%20sports%20facility%20with%20athletes.png"
        theme_icons = ["⚽", "🏆", "💪"]
        theme_content = {
            "hero_title": "Achieve Athletic Excellence",
            "hero_subtitle": "Train with the best, become your best",
            "about_title": "Your Sports Destination",
            "about_content": "We provide comprehensive training programs and facilities for athletes of all levels to reach their full potential.",
            "services": ["Professional Training", "Sports Programs", "Fitness Assessment"]
        }
    
    # Generate contextual content based on user input (first 100 chars)
    context = user_text[:100] if len(user_text) > 20 else "A modern, professional website"
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-30px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        .animate-fade {{ animation: fadeIn 0.8s ease-out; }}
        .animate-slide {{ animation: slideIn 0.8s ease-out; }}
    </style>
</head>
<body class="bg-gradient-to-br {color_scheme} min-h-screen">
    <nav class="bg-white/90 backdrop-blur-md shadow-lg sticky top-0 z-50">
        <div class="container mx-auto px-6 py-4">
            <div class="flex justify-between items-center">
                <h1 class="text-2xl font-bold bg-gradient-to-r {color_scheme} bg-clip-text text-transparent">{title}</h1>
                <div class="space-x-6">
                    <a href="#home" class="text-gray-700 hover:text-gray-900 font-medium transition">Home</a>
                    <a href="#features" class="text-gray-700 hover:text-gray-900 font-medium transition">Features</a>
                    <a href="#about" class="text-gray-700 hover:text-gray-900 font-medium transition">About</a>
                    <a href="#contact" class="text-gray-700 hover:text-gray-900 font-medium transition">Contact</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section id="home" class="container mx-auto px-6 py-24 animate-fade">
        <div class="text-center max-w-4xl mx-auto">
            <h2 class="text-6xl md:text-7xl font-bold text-white mb-6 drop-shadow-lg">Welcome to {title}</h2>
            <p class="text-xl md:text-2xl text-white/90 mb-10 leading-relaxed">{description}</p>
            <div class="flex gap-4 justify-center">
                <button class="bg-white text-gray-900 px-8 py-4 rounded-lg font-semibold hover:bg-gray-100 transition transform hover:scale-105 shadow-xl">Get Started</button>
                <button class="bg-white/20 backdrop-blur-sm text-white px-8 py-4 rounded-lg font-semibold hover:bg-white/30 transition border-2 border-white/50">Learn More</button>
            </div>
        </div>
    </section>
    
    <section id="features" class="container mx-auto px-6 py-20 bg-white">
        <h2 class="text-4xl md:text-5xl font-bold text-center mb-16 text-gray-800">Why Choose Us</h2>
        <div class="grid md:grid-cols-3 gap-8">
            <div class="bg-gradient-to-br from-gray-50 to-gray-100 p-8 rounded-xl shadow-lg hover:shadow-2xl transition transform hover:-translate-y-2">
                <div class="text-5xl mb-4">✨</div>
                <h3 class="text-2xl font-bold mb-4 text-gray-800">{features[0]}</h3>
                <p class="text-gray-600">Experience excellence in every detail with our premium services.</p>
            </div>
            <div class="bg-gradient-to-br from-gray-50 to-gray-100 p-8 rounded-xl shadow-lg hover:shadow-2xl transition transform hover:-translate-y-2">
                <div class="text-5xl mb-4">🚀</div>
                <h3 class="text-2xl font-bold mb-4 text-gray-800">{features[1]}</h3>
                <p class="text-gray-600">Innovative solutions designed to meet your unique needs.</p>
            </div>
            <div class="bg-gradient-to-br from-gray-50 to-gray-100 p-8 rounded-xl shadow-lg hover:shadow-2xl transition transform hover:-translate-y-2">
                <div class="text-5xl mb-4">💎</div>
                <h3 class="text-2xl font-bold mb-4 text-gray-800">{features[2]}</h3>
                <p class="text-gray-600">Quality and reliability you can trust for all your requirements.</p>
            </div>
        </div>
    </section>
    
    <section id="about" class="container mx-auto px-6 py-20 bg-gradient-to-br from-gray-50 to-gray-100">
        <div class="max-w-4xl mx-auto">
            <h2 class="text-4xl md:text-5xl font-bold text-center mb-12 text-gray-800">About {title}</h2>
            <div class="grid md:grid-cols-2 gap-12 items-center">
                <div>
                    <p class="text-lg text-gray-700 mb-6 leading-relaxed">
                        We are dedicated to providing exceptional services and creating memorable experiences for our clients. 
                        With a commitment to excellence and innovation, we strive to exceed expectations in everything we do.
                    </p>
                    <p class="text-lg text-gray-700 leading-relaxed">
                        Our team of experts works tirelessly to deliver solutions that are both practical and transformative, 
                        ensuring that every interaction adds value to your journey.
                    </p>
                </div>
                <div class="bg-white p-8 rounded-xl shadow-xl">
                    <div class="grid grid-cols-2 gap-6">
                        <div class="text-center">
                            <div class="text-4xl font-bold bg-gradient-to-r {color_scheme} bg-clip-text text-transparent mb-2">100+</div>
                            <div class="text-gray-600">Happy Clients</div>
                        </div>
                        <div class="text-center">
                            <div class="text-4xl font-bold bg-gradient-to-r {color_scheme} bg-clip-text text-transparent mb-2">5★</div>
                            <div class="text-gray-600">Rating</div>
                        </div>
                        <div class="text-center">
                            <div class="text-4xl font-bold bg-gradient-to-r {color_scheme} bg-clip-text text-transparent mb-2">24/7</div>
                            <div class="text-gray-600">Support</div>
                        </div>
                        <div class="text-center">
                            <div class="text-4xl font-bold bg-gradient-to-r {color_scheme} bg-clip-text text-transparent mb-2">10+</div>
                            <div class="text-gray-600">Years Experience</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <section id="contact" class="container mx-auto px-6 py-20 bg-white">
        <h2 class="text-4xl md:text-5xl font-bold text-center mb-12 text-gray-800">Get In Touch</h2>
        <div class="max-w-2xl mx-auto">
            <div class="bg-gradient-to-br {color_scheme} p-12 rounded-2xl shadow-2xl">
                <form class="space-y-6">
                    <div>
                        <input type="text" placeholder="Your Name" class="w-full px-6 py-4 rounded-lg border-0 focus:ring-4 focus:ring-white/50 transition bg-white/90 backdrop-blur-sm text-gray-800 placeholder-gray-500">
                    </div>
                    <div>
                        <input type="email" placeholder="Your Email" class="w-full px-6 py-4 rounded-lg border-0 focus:ring-4 focus:ring-white/50 transition bg-white/90 backdrop-blur-sm text-gray-800 placeholder-gray-500">
                    </div>
                    <div>
                        <textarea placeholder="Your Message" class="w-full px-6 py-4 rounded-lg border-0 focus:ring-4 focus:ring-white/50 transition bg-white/90 backdrop-blur-sm text-gray-800 placeholder-gray-500" rows="5"></textarea>
                    </div>
                    <button type="submit" class="w-full bg-white text-gray-900 py-4 rounded-lg font-bold hover:bg-gray-100 transition transform hover:scale-105 shadow-xl">Send Message</button>
                </form>
            </div>
        </div>
    </section>
    
    <footer class="bg-gray-900 text-white py-12">
        <div class="container mx-auto px-6">
            <div class="grid md:grid-cols-3 gap-8 mb-8">
                <div>
                    <h3 class="text-2xl font-bold mb-4">{title}</h3>
                    <p class="text-gray-400">Creating exceptional experiences for our clients.</p>
                </div>
                <div>
                    <h4 class="text-lg font-semibold mb-4">Quick Links</h4>
                    <ul class="space-y-2 text-gray-400">
                        <li><a href="#home" class="hover:text-white transition">Home</a></li>
                        <li><a href="#features" class="hover:text-white transition">Features</a></li>
                        <li><a href="#about" class="hover:text-white transition">About</a></li>
                        <li><a href="#contact" class="hover:text-white transition">Contact</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="text-lg font-semibold mb-4">Connect</h4>
                    <div class="flex space-x-4">
                        <a href="#" class="text-gray-400 hover:text-white transition">Facebook</a>
                        <a href="#" class="text-gray-400 hover:text-white transition">Twitter</a>
                        <a href="#" class="text-gray-400 hover:text-white transition">Instagram</a>
                    </div>
                </div>
            </div>
            <div class="border-t border-gray-800 pt-8 text-center text-gray-400">
                <p>&copy; 2024 {title}. All rights reserved.</p>
            </div>
        </div>
    </footer>
    
    <script>
        // Smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    return f"I've created a beautiful {website_type} website for you.\n\n```html\n{html_template}\n```"

# Function to process text using LLM with retry and fallback
def process_with_llm(text, html_content="", max_retries=3):
    system_prompt = "You are a helpful software engineer, you will write HTML with TailWind CSS code embedded into it.Give content from html start tag to end tag. Implement proper scroll for the webpage. Even if the user has given very little description, use your creatvity and make a good complete and large webpage.You will have to write the code for the frontend. If there is a website already, the code of that website will also be given to you. Make top and side pop up navigation bars,  proper scroll functionality, back button, etc. Use animations for all photos, text, buttons, everything. Have some background theme and colors do not keep it blank white. Have a lot of content on the webpage, do not make a plain simple empty website. Include images to make the website look good wherever neceessary. Maximum 10 images. To include images use https://image.pollinations.ai/prompt/<imagedescription>.png for the image url. Write code such that the bottom 10% of the image is not visble, as that part contains watermark. You can use image for background as well, make sure it is not repeating and is rendered properly. Crop the images into different size and shapes. Your response should consist of 2 parts seperated by 2 new lines. first part will be A reply message to the user in the input language saying how you have completed the work. 2nd part will be the code"
    
    reference_text = f"Reference Code: {html_content} You can refer the given code for ideas." if html_content else "Create a new website from scratch."
    user_prompt = f"User Instruction: {text}. {reference_text}"
    
    # Try with retry logic
    for attempt in range(max_retries):
        try:
            response = LLMclient.chat.completions.create(
                model=LLM_PRIMARY,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=60
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    # Extract retry delay if available
                    wait_time = 60  # Default wait time
                    if "retry in" in error_str.lower():
                        try:
                            # Try to extract wait time from error message
                            import re
                            match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                            if match:
                                wait_time = int(float(match.group(1))) + 5
                        except:
                            pass
                    
                    st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("Rate limit exceeded. Using template fallback.")
                    return generate_html_template(text)
            else:
                # Other errors - try fallback immediately
                st.warning(f"API error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    st.info("Using template fallback due to API errors.")
                    return generate_html_template(text)
    
    # Final fallback
    return generate_html_template(text)

# Language code mapping
language_mapping = {
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",

    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Odia": "od-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN"
}

# Function to process user input (text or transcript)
def process_user_input(user_text, language_code):
    """Process user input and generate HTML"""
    if not user_text or user_text.strip() == "":
        st.warning("Please provide some input.")
        return
    
    st.write(f"**User Input 🙋:** {user_text}")
    
    # Fetch embeddings and nearest neighbors
    with st.spinner("Searching for similar designs..."):
        results = get_embeddings_and_neighbors(user_text)
    
    # Get HTML content from results or use empty string
    html_content = ""
    if results and len(results) > 0 and results[0]:
        try:
            html_content = results[0][2] if len(results[0]) > 2 else ""
            if html_content:
                st.info(f"Found similar design: {results[0][1] if len(results[0]) > 1 else 'N/A'}")
        except (IndexError, TypeError):
            html_content = ""
    
    # Process with LLM
    with st.spinner("Generating your website..."):
        sarvam_message = process_with_llm(user_text, html_content)
    
    # Save log
    with open("log.txt", "w", encoding='utf-8') as f:
        f.write(sarvam_message)
    
    # Display model response
    if sarvam_message:
        response_parts = sarvam_message.split('\n\n')
        if len(response_parts) > 0:
            st.write(f"**Model Response 🤖:** {response_parts[0]}")
        
        # Extract HTML code
        html_code = ""
        start_marker = "```html"
        end_marker = "```"
        
        # Try to find HTML code block
        start_index = sarvam_message.find(start_marker)
        if start_index != -1:
            end_index = sarvam_message.find(end_marker, start_index + len(start_marker))
            if end_index != -1:
                html_code = sarvam_message[start_index + len(start_marker):end_index].strip()
        else:
            # If no code block markers, try to extract HTML directly
            # Look for <html> tag
            html_start = sarvam_message.find("<html>")
            html_end = sarvam_message.rfind("</html>")
            if html_start != -1 and html_end != -1:
                html_code = sarvam_message[html_start:html_end + 7]
        
        # Save HTML file
        if html_code:
            with open("generated_app.html", "w", encoding='utf-8') as f:
                f.write(html_code)
            st.success("✓ HTML file generated successfully!")
            
            # Display preview
            st.subheader("Preview:")
            st.components.v1.html(html_code, height=600, scrolling=True)
            
            # Open in browser button
            if st.button("Open in Browser"):
                subprocess.run(["start", "generated_app.html"], shell=True)
        else:
            st.warning("Could not extract HTML code from response. Check log.txt for full response.")
            st.text_area("Full Response:", sarvam_message, height=300)
    
    # Generate audio response
    try:
        with st.spinner("Generating audio response..."):
            response_text = response_parts[0][:490] if len(response_parts) > 0 else user_text[:490]
            payload = {
                "inputs": [response_text],
                "target_language_code": language_code,
                "speaker": "meera",
                "pitch": 0.2,
                "pace": 1.1,
                "loudness": 0.8,
                "enable_preprocessing": True,
                "model": "bulbul:v1",
                "speech_sample_rate": 8000
            }
            response = requests.request("POST", sarvamurl, json=payload, headers=sarvamheaders)
            audio_data = response.json()
            if "audios" in audio_data and audio_data["audios"]:
                audio_bytes = base64.b64decode(audio_data["audios"][0])
                st.audio(audio_bytes, format="audio/wav", autoplay=False)
    except Exception as e:
        st.warning(f"Could not generate audio: {e}")

# Streamlit UI
st.title("🎨 Design Companion")
st.markdown("Create beautiful websites with AI - Use voice or text input!")

# Move language selection to sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_language = st.selectbox("Select Language:", 
                                  list(language_mapping.keys()), 
                                  index=0)
    language_code = language_mapping[selected_language]
    
    st.markdown("---")
    st.markdown("### Input Method")
    input_method = st.radio("Choose input type:", ["Text Input", "Voice Input"], index=0)

# Main input area
if input_method == "Text Input":
    user_text = st.text_area("Enter your website description:", 
                            placeholder="E.g., Create a modern restaurant website with menu, gallery, and contact form...",
                            height=100)
    
    if st.button("🚀 Generate Website", type="primary"):
        if user_text and user_text.strip():
            process_user_input(user_text.strip(), language_code)
        else:
            st.warning("Please enter some text first!")
else:
    audio_value = st.audio_input("Record a voice message")
    
    if audio_value:
        if subscription_key:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_value, subscription_key, language_code)
            
            if transcript and not transcript.startswith("Error"):
                process_user_input(transcript, language_code)
            else:
                st.error(f"Transcription failed: {transcript}")
        else:
            st.error("API Subscription Key not found. Please check your configuration.")
