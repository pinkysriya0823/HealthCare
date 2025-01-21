from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit

from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from services import ocr_service
from langchain_core.messages import AIMessage
from datetime import timedelta,datetime
from flask import flash
import threading
import pygame
import os
import time
from flask import jsonify 
from gtts import gTTS
from threading import Thread, Event
from apscheduler.schedulers.background import BackgroundScheduler
from bson.objectid import ObjectId
import logging
import speech_recognition as sr
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import tempfile


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key
socketio = SocketIO(app)

# MongoDB Setup
app.config['MONGO_URI'] = 'mongodb://localhost:27017/healthcare'  # Adjust URI if using MongoDB Atlas
mongo = PyMongo(app)
logging.basicConfig(level=logging.DEBUG)

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check user credentials in MongoDB
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):  # Validate hashed password
            session['user_id'] = str(user['_id'])  # Store user ID in session
            return redirect(url_for('dashboard'))
        else:
            error= "Invalid credentials, try again"
    
    return render_template('signin.html', error=error)

# Sign-Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)  # Hash the password
        
        existing_user = mongo.db['users'].find_one({'email': email})  # Correct way to access the 'users' collection
        if existing_user:
            error = "Email already exists! Please choose a different one."
        # Insert new user into MongoDB
        else:
            # Insert new user into MongoDB
            mongo.db['users'].insert_one({'email': email, 'password': hashed_password})
            return redirect(url_for('signin'))  # Redirect to login page after successful signup
    return render_template('signup.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    
    user_id = session['user_id']
    
    # Fetch the user's health data from MongoDB
    health_data = mongo.db.health_data.find({'user_id': user_id})  # Fetch only the health data for the logged-in user
    
    # Convert the MongoDB cursor to a list of dictionaries
    health_data_list = list(health_data)
    # Initialize variables for average calculation
    avg_blood_pressure_systolic = avg_blood_pressure_diastolic = avg_heart_rate = avg_oxygen_saturation = 0
    avg_blood_sugar = avg_cholesterol = avg_body_temperature = avg_sleep_duration = avg_exercise_duration = avg_calories_burned = 0
    
    # Calculate the averages for each metric if there is data
    if health_data_list:
        # Blood Pressure
        avg_blood_pressure_systolic = round(
            sum([int(data['blood_pressure']['systolic']) for data in health_data_list if data['blood_pressure'].get('systolic')]) / len(health_data_list),
            1
        )
        avg_blood_pressure_diastolic = round(
            sum([int(data['blood_pressure']['diastolic']) for data in health_data_list if data['blood_pressure'].get('diastolic')]) / len(health_data_list),
            1
        )

        # Heart Rate
        avg_heart_rate = round(
            sum([int(data['heart_rate']) for data in health_data_list if data.get('heart_rate')]) / len(health_data_list),
            1
        )

        # Oxygen Saturation (Handle empty or invalid data)
        avg_oxygen_saturation = round(
            sum([float(data['oxygen_saturation']) for data in health_data_list if data.get('oxygen_saturation') and data['oxygen_saturation'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Blood Sugar
        avg_blood_sugar = round(
            sum([float(data['blood_sugar']) for data in health_data_list if data.get('blood_sugar') and data['blood_sugar'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Cholesterol Level
        avg_cholesterol = round(
            sum([float(data['cholesterol_level']) for data in health_data_list if data.get('cholesterol_level') and data['cholesterol_level'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Body Temperature
        avg_body_temperature = round(
            sum([float(data['temperature']) for data in health_data_list if data.get('temperature') and data['temperature'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Sleep Duration
        avg_sleep_duration = round(
            sum([float(data['sleep_duration']) for data in health_data_list if data.get('sleep_duration') and data['sleep_duration'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Exercise Duration
        avg_exercise_duration = round(
            sum([float(data['exercise_duration']) for data in health_data_list if data.get('exercise_duration') and data['exercise_duration'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

        # Calories Burned
        avg_calories_burned = round(
            sum([float(data['calories_burned']) for data in health_data_list if data.get('calories_burned') and data['calories_burned'].replace('.', '', 1).isdigit()]) / len(health_data_list),
            1
        )

    else:
        # Set default values if no data is available
        avg_blood_pressure_systolic = avg_blood_pressure_diastolic = avg_heart_rate = avg_oxygen_saturation = 0
        avg_blood_sugar = avg_cholesterol = avg_body_temperature = avg_sleep_duration = avg_exercise_duration = avg_calories_burned = 0

    return render_template('dashboard.html', 
                           health_data=health_data_list, 
                           avg_blood_pressure_systolic=avg_blood_pressure_systolic,
                           avg_blood_pressure_diastolic=avg_blood_pressure_diastolic,
                           avg_heart_rate=avg_heart_rate,
                           avg_oxygen_saturation=avg_oxygen_saturation,
                           avg_blood_sugar=avg_blood_sugar,
                           avg_cholesterol=avg_cholesterol,
                           avg_body_temperature=avg_body_temperature,
                           avg_sleep_duration=avg_sleep_duration,
                           avg_exercise_duration=avg_exercise_duration,
                           avg_calories_burned=avg_calories_burned)
 



# Add Health Report Route
@app.route('/add_report', methods=['GET', 'POST'])
def add_report():
    # Define database name and collection name
    database_name = "healthcare"  # Replace with your database name
    collection_name = "health_data"  # Replace with your collection name
    if request.method == 'POST':
        # Process uploaded report (using OCR service)
        file = request.files['report']
        # Analyze the report using the OCR service
        
        analysis = ocr_service.analyze_report(file)
        # Check if analysis is an AIMessage and serialize it
        if isinstance(analysis, AIMessage):
            analysis_data = analysis.content 
        else:
            # If analysis is not AIMessage, store it as is
            analysis_data = analysis
        # Add the report analysis to the database
    
        return render_template('add_report.html', analysis=analysis_data)  # Render the page with analysis
    return render_template('add_report.html')  # Render the page with the form to upload the report
# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorstore_file = "vectorstore_med_part2.index"
if os.path.exists(vectorstore_file):
    logging.info("Loading existing vector store...")
    vectorstore = FAISS.load_local(vectorstore_file, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
else:
    logging.error("Vector store file does not exist.")

# Initialize the language model
llm = ChatGroq(
    groq_api_key='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',
    temperature=0.2,
    max_tokens=3000,
    model_kwargs={"top_p": 1}
)

# Define the chat prompt template
template = """
<|context|>
You are a professional medical doctor who provides accurate and empathetic advice based on the patient's symptoms and health concerns. Your primary focus is on offering tailored solutions that address the patient's condition effectively.

Responsibilities:
- Accept user-provided symptoms to suggest possible health improvements.
- Diagnose the patient accurately and offer direct, truthful answers.
- Recommend appropriate medications, diet plans, and exercises tailored to the user's condition.
- Use AI-driven insights to propose personalized exercise and diet plans aligned with the user's health status.
- Share tips for maintaining a healthy lifestyle based on data and symptom analysis.

Formatting Guidelines:
- Provide responses in a proper format with clear spacing and indentations.
- Use separate lines for each point or suggestion to improve readability.
- Avoid presenting multiple points on the same line.
- Keep the language simple and empathetic while maintaining professionalism.
- Ensure the conversation is concise but sufficiently detailed to address the query effectively.

</s>
<|user|>
{query}
</s>
<|assistant|>


"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define the response function
def get_response(user_input,conversation_context):
    logging.info(f"Getting response for user input: {user_input}")
    try:
        full_prompt = "Chat History: " + conversation_context + " User Input: " + user_input

        result = rag_chain.invoke(full_prompt)
        time.sleep(1)  # Simulate processing delay
        return result
    except Exception as e:
        logging.error(f"Error while fetching response: {e}")
        return "Sorry, something went wrong. Please try again."


# Chatbot Route (Symptom Analysis)
@app.route('/chatbot')
def chatbot():
    
    return render_template('chatbot01.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']  # Get user input from form
    user_id = session.get('user_id')  # Assuming user_id is stored in the session after login

    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403
    # Fetch the user's chat history document or create one if it doesn't exist
    chat_history_collection = mongo.db.chat_history
    chat_record = chat_history_collection.find_one({'user_id': user_id})

    if not chat_record:
        chat_record = {'user_id': user_id, 'history': []}
        chat_history_collection.insert_one(chat_record)

    # Append the user's message to the history
    chat_record['history'].append({'user': user_input})
    # Prepare the conversation context for the bot to use
    conversation_context = " ".join([entry.get('user', '') + " " + entry.get('bot', '') for entry in chat_record['history']])

    # Get response
    response = get_response(user_input, conversation_context)
    
    # Append the chatbot's response to the history
    chat_record['history'].append({'bot': response})

    # Update the chat history in the database
    chat_history_collection.update_one(
        {'user_id': user_id},
        {'$set': {'history': chat_record['history']}}
    )

    return jsonify({'response': response})

@app.route('/history', methods=['GET'])
def chat_history():
    user_id = session.get('user_id')  # Assuming user_id is stored in the session after login
    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403

    # Retrieve the chat history from MongoDB
    chat_history_collection = mongo.db.chat_history
    chat_record = chat_history_collection.find_one({'user_id': user_id})

    if not chat_record:
        return jsonify({'history': []})

    return jsonify({'history': chat_record['history']})
    
@app.route('/clear_history', methods=['POST'])
def clear_history():
    user_id = session.get('user_id')  # Get the user ID from session
    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403

    # Access the chat history collection in MongoDB
    chat_history_collection = mongo.db.chat_history
    
    # Delete the chat record for the logged-in user
    result = chat_history_collection.delete_one({'user_id': user_id})

    if result.deleted_count > 0:
        return jsonify({'message': 'Chat history cleared.'})
    else:
        return jsonify({'error': 'No chat history found.'}), 404


@app.route('/voice-input', methods=['POST'])
def voice_input():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Get the audio file from the form
    audio_file = request.files['audio']

    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=True) as temp_audio:
        audio_file.save(temp_audio.name)

        # Recognize the speech using Google Web Speech API (you can replace this with other services)
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio)
                return jsonify({'user_input': user_input})
            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand the audio.'})
            except sr.RequestError as e:
                return jsonify({'error': f'Could not request results; {e}'})


@app.route('/add_health_data', methods=['GET', 'POST'])
def add_health_data():
    if request.method == 'POST':
        # Get form data
        blood_pressure_systolic = request.form.get('blood_pressure_systolic')
        blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
        heart_rate = request.form.get('heart_rate')
        temperature = request.form.get('temperature')
        respiratory_rate = request.form.get('respiratory_rate')
        oxygen_saturation = request.form.get('oxygen_saturation')
        blood_sugar = request.form.get('blood_sugar')
        cholesterol_level = request.form.get('cholesterol_level')
        steps_taken = request.form.get('steps_taken')
        exercise_duration = request.form.get('exercise_duration')
        calories_burned = request.form.get('calories_burned')
        sleep_duration = request.form.get('sleep_duration')
        sleep_quality = request.form.get('sleep_quality')
        meal_timing = request.form.get('meal_timing')
        water_intake = request.form.get('water_intake')
        chronic_diseases = request.form.get('chronic_diseases')
        recent_diseases = request.form.get('recent_diseases')
        allergies = request.form.get('allergies')
        medications = request.form.get('medications')
        stress_level = request.form.get('stress_level')
        mood = request.form.get('mood')
        mental_health_conditions = request.form.get('mental_health_conditions')
        air_quality = request.form.get('air_quality')
        weather_conditions = request.form.get('weather_conditions')
        
        user_id = session['user_id']  # Assuming the user is logged in
        
        # Insert the data into MongoDB
        mongo.db.health_data.insert_one({
            'user_id': user_id,
            'blood_pressure': {
                'systolic': blood_pressure_systolic,
                'diastolic': blood_pressure_diastolic
            },
            'heart_rate': heart_rate,
            'temperature': temperature,
            'respiratory_rate': respiratory_rate,
            'oxygen_saturation': oxygen_saturation,
            'blood_sugar': blood_sugar,
            'cholesterol_level': cholesterol_level,
            'steps_taken': steps_taken,
            'exercise_duration': exercise_duration,
            'calories_burned': calories_burned,
            'sleep_duration': sleep_duration,
            'sleep_quality': sleep_quality,
            'meal_timing': meal_timing,
            'water_intake': water_intake,
            'chronic_diseases': chronic_diseases,
            'recent_diseases': recent_diseases,
            'allergies': allergies,
            'medications': medications,
            'stress_level': stress_level,
            'mood': mood,
            'mental_health_conditions': mental_health_conditions,
            'air_quality': air_quality,
            'weather_conditions': weather_conditions,
            'timestamp': datetime.utcnow()  # Optional: add timestamp for when the data was added
        })
        
        return redirect(url_for('dashboard'))  # Redirect to dashboard after successful submission
    
    return render_template('add_health_data.html')  # Render the form to add health data



# Counter for repeating alarm sound


alarm_counter = 0
max_repeats = 6  

# Global stop event for controlling the reminder playback
stop_event = threading.Event()
scheduler = BackgroundScheduler()

def play_alarm(reminder_text, reminder_id):
    """Play the alarm sound and update the reminder status to 'missed' if needed."""
    global alarm_counter
    stop_event.clear()
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Check if pygame mixer is successfully initialized
        if not pygame.mixer.get_init():
            print("Error: Pygame mixer is not initialized.")
            return
        tts = gTTS(reminder_text, lang='en')
        tts.save("reminder.mp3")  # Save it as reminder.mp3
        alarm_sound = pygame.mixer.Sound('alarm1.mp3')  # The regular alarm sound
        reminder_sound = pygame.mixer.Sound('reminder.mp3')  

        while alarm_counter < max_repeats and not stop_event.is_set():
            try:
                socketio.emit('play_alarm_sound', {'message': reminder_text, 'reminder_id': reminder_id})
                # Play the alarm sound first
                alarm_sound.play()

                # Wait for the alarm to finish
                time.sleep(alarm_sound.get_length())

                # Stop the alarm sound after it finishes
                alarm_sound.stop()
                print("Alarm sound finished playing.")

                # Play the reminder sound after the alarm
                reminder_sound.play()

                # Wait for the reminder sound to finish
                time.sleep(reminder_sound.get_length())

                # Stop the reminder sound after it finishes
                reminder_sound.stop()
                print("Reminder sound finished playing.")

                # Play the alarm sound
                alarm_sound.play()

                # Let the alarm sound play for 9 seconds (adjust as needed)
                time.sleep(alarm_sound.get_length())   # Match the alarm's actual duration

                # Stop the alarm sound
                alarm_sound.stop()

            except pygame.error as e:
                print(f"Error loading or playing sound: {e}")
                return

            alarm_counter += 1  # Increment the repeat counter
            print("391")
            # Check if the alarm has been triggered 6 times and hasn't been marked as completed
            if alarm_counter >= max_repeats:
                # Update the status in the database to 'missed'
                mongo.db.reminders.update_one(
                    {'_id': ObjectId(reminder_id)},
                    {'$set': {'status': 'missed'}}
                )
                print(f"Reminder {reminder_text} marked as missed.")
                alarm_counter=0
                break

    except Exception as e:
        print(f"Error playing alarm: {e}")  

# SocketIO events for communication with the front-end
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
def check_reminders():
    """Checks pending reminders and triggers alarm if reminder time is met."""
    current_time = datetime.now()
    current_time = current_time + timedelta(hours=5, minutes=30)

    reminders = mongo.db.reminders.find({'status': 'pending'})
    print(f"Current Time (UTC): {current_time.isoformat()}")

    for reminder in reminders:
        reminder_time_str = reminder['reminder_time']
        reminder_time_obj = datetime.fromisoformat(reminder_time_str)  # Convert string to datetime object
        print(f"Reminder Time: {reminder_time_obj.isoformat()}")

        if reminder_time_obj <= current_time:
            # Trigger the alarm (for demonstration, print reminder text)
            print(f"Reminder: {reminder['reminder_text']} at {reminder_time_str}")
            play_alarm(reminder['reminder_text'],str(reminder['_id']))

                        # Handle repeat functionality
            new_reminder_time = None
            if reminder['repeat_frequency'] == 'weekly':
                new_reminder_time = reminder_time_obj + timedelta(weeks=1)
            elif reminder['repeat_frequency'] == 'daily':
                new_reminder_time = reminder_time_obj + timedelta(days=1)
            elif reminder['repeat_frequency'] == 'monthly':
                new_reminder_time = reminder_time_obj + timedelta(weeks=4)  # Approximation of a month

            if new_reminder_time:
                # Create a new reminder with the same details but a new reminder time
                new_reminder = {
                    'user_id': reminder['user_id'],
                    'reminder_text': reminder['reminder_text'],
                    'reminder_time': new_reminder_time.isoformat(),
                    'repeat_frequency': reminder['repeat_frequency'],
                    'status': 'pending'
                }

                # Insert the new reminder into the database
                mongo.db.reminders.insert_one(new_reminder)

                print(f"New reminder created at {new_reminder_time.isoformat()}")

# Add the function to the scheduler
scheduler.add_job(func=check_reminders, trigger="interval", minutes=1)  # Check reminders every minute

# Start the scheduler
scheduler.start()

# Reminder Routes
@app.route('/reminders')
def reminders():
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    
    user_id = session['user_id']
    reminders = mongo.db.reminders.find({'user_id': user_id})
    reminders_list = list(reminders)
    
    return render_template('reminders.html', reminders=reminders_list)



@app.route('/mark_reminder_done/<reminder_id>', methods=['POST'])
def mark_reminder_done(reminder_id):
    """Mark a reminder as completed."""
    global alarm_counter  # Make sure to reset the counter on completion
    stop_event.set()  # Stop the alarm thread
    alarm_counter = 0  # Reset the alarm counter

    try:
        mongo.db.reminders.update_one(
            {'_id': ObjectId(reminder_id)},
            {'$set': {'status': 'completed'}}
        )
        logging.info(f"Reminder {reminder_id} marked as completed.")
    except Exception as e:
        logging.error(f"Error marking reminder as completed: {e}")

    return redirect(url_for('reminders'))

@app.route('/set_reminder', methods=['GET', 'POST'])
def set_reminder():
    if request.method == 'POST':
        reminder_text = request.form['reminder_text']
        reminder_time = request.form['reminder_time']
        repeat_frequency = request.form.get('repeat_frequency', 'none')  # Default to 'none'
        user_id = session['user_id']
        
        reminder = {
            'user_id': user_id,
            'reminder_text': reminder_text,
            'reminder_time': reminder_time,
            'repeat_frequency': repeat_frequency,
            'status': 'pending'
        }
        
        mongo.db.reminders.insert_one(reminder)
        return redirect(url_for('reminders'))
    
    return redirect(url_for('reminders'))

@app.route('/update_reminder/<reminder_id>', methods=['GET', 'POST'])
def update_reminder(reminder_id):
    reminder_object_id = ObjectId(reminder_id)

    reminder = mongo.db.reminders.find_one({'_id': reminder_object_id})
    
    if request.method == 'POST':
        reminder_text = request.form['reminder_text']
        reminder_time = request.form['reminder_time']
        repeat_frequency = request.form['repeat_frequency']  # Get the repeat frequency value
        
        # Update the reminder document with the new data
        mongo.db.reminders.update_one(
            {'_id': reminder_object_id},
            {'$set': {'reminder_text': reminder_text, 'reminder_time': reminder_time, 'repeat_frequency': repeat_frequency}}
        )

        return redirect(url_for('reminders'))  # Redirect to the list of reminders
    
    return render_template('update_reminder.html', reminder=reminder)

@app.route('/delete_reminder/<reminder_id>', methods=['POST'])
def delete_reminder(reminder_id):
    reminder_object_id = ObjectId(reminder_id)
    mongo.db.reminders.delete_one({'_id': reminder_object_id})
    return redirect(url_for('reminders'))

def get_last_health_data(user_id, max_entries=7):
    """
    Fetch the last N entries of health data for a given user.
    """
    health_data_collection = mongo.db.health_data
    # Fetch the latest health data for the user, limited by max_entries
    health_entries = list(
        health_data_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(max_entries)
    )
    return health_entries


#---------------------------------------------


from flask import send_file

@app.route('/get_reminder_audio/<reminder_id>')
def get_reminder_audio(reminder_id):
    # Generate the reminder TTS audio file
    reminder = mongo.db.reminders.find_one({'_id': ObjectId(reminder_id)})
    if not reminder:
        return "Reminder not found", 404
    
    reminder_text = reminder['reminder_text']
    tts = gTTS(reminder_text, lang='en')
    audio_file_path = f"reminder_{reminder_id}.mp3"
    tts.save(audio_file_path)
    
    return send_file(audio_file_path, mimetype="audio/mpeg")




def get_response1(user_input):
    logging.info(f"Getting response for user input: {user_input}")
    try:

        result = rag_chain.invoke(user_input)
        time.sleep(1)  # Simulate processing delay
        return result
    except Exception as e:
        logging.error(f"Error while fetching response: {e}")
        return "Sorry, something went wrong. Please try again."

def generate_recommendations(health_data):
    """
    Generate recommendations for diet, exercise, and lifestyle based on health data.
    """
    # Combine health data into a query string for the LLM
    health_summary = "\n".join(
        f"Entry {i+1}: {entry}"
        for i, entry in enumerate(health_data)
    )
    
    query = f"""
    Based on the following health data:
    {health_summary}
    
    Provide detailed recommendations for:
    - Diet
    - Exercise
    - Lifestyle improvements
    use "/n" to represent next line
    """
    
    return get_response1(query)  # No prior chat context needed for health history


@app.route('/health_recommendations', methods=['GET'])
def health_recommendations():
    user_id = session.get('user_id')  # Get the logged-in user ID from the session

    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403

    # Fetch the last 7 (or fewer) health entries
    health_data = get_last_health_data(user_id)
    
    if not health_data:
        return jsonify({'error': 'No health data found for the user.'}), 404

    # Generate recommendations using the LLM
    recommendations = generate_recommendations(health_data)
    #return jsonify({'recommendations': recommendations})
    if isinstance(recommendations, str):
        recommendations = {"General Recommendations": recommendations}
    return render_template('recommendations.html', recommendations=recommendations)
    

@app.route('/logout')
def logout():
    # Clear user session
    session.clear()  # Ensure the session is cleared

    # Redirect to login page (adjust the route as per your app's login page)
    return redirect('/')
def reset_alarm_counter():
    global alarm_counter
    alarm_counter = 0 

ssl_context = ('/home/ubuntu/certs/server.crt', '/home/ubuntu/certs/private.key')

if __name__ == '__main__':

    #app.run(ssl_context=('/home/ubuntu/certs/server.crt', '/home/ubuntu/certs/private.key'), host='0.0.0.0', port=5000, debug=False)
    socketio.run(app, ssl_context=ssl_context, host='0.0.0.0', port=5000, debug=False)



