# Use a slim version of Python to keep the image small
FROM python:3.11-slim

# Set the folder where our code will live inside the "box"
WORKDIR /app

# Copy only the requirements first (helps with faster building)
COPY requirements.txt .

# Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the box
COPY . .

# Open the port that FastAPI uses
EXPOSE 8000

# The command to start your environment server
# Using the main function we defined in app.py
CMD ["python", "server/app.py"]