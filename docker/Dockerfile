
# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file from the root of the build context
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project directory (the build context) into the container
COPY . .

# 6. Make port 8000 available to the world outside this container
EXPOSE 8000

# 7. Define the command to run the app using Gunicorn
# The command is `gunicorn --bind 0.0.0.0:8000 app:app`
#   - `--bind 0.0.0.0:8000`: Binds the server to all network interfaces on port 8000.
#   - `app:app`: Tells Gunicorn to look for the `app` object in the `app.py` file.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]