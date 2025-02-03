# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Coreferee for English
python -m coreferee install en

echo "Setup completed successfully!"
