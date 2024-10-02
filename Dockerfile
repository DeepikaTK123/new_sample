# Use a specific, stable Node.js runtime as a parent image
FROM node:18-alpine

# Set the working directory
WORKDIR /app

# Copy package.json and yarn.lock
COPY package*.json ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Expose port 3000 (uncomment if your app listens on this port)
EXPOSE 3000

# Define the command to run the application
CMD ["yarn", "start"]


