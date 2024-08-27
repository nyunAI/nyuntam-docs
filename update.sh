#!/bin/bash

CWD=$(pwd)
echo $CWD

PUBLISH=$CWD/publish

# Clone the repository's gh-pages branch into the "publish" directory
echo "🚀 Cloning gh-pages branch into publish directory..."
git clone -b gh-pages https://github.com/nyunai/nyuntam-docs.git $PUBLISH

# Clear out existing content in the "publish" directory
echo "🚀 Clearing existing content in the publish directory..."
rm -rf $PUBLISH/*

# Create .nojekyll file to ensure GitHub Pages does not process as a Jekyll site
echo "🚀 Creating .nojekyll file..."
touch $PUBLISH/.nojekyll

# Install Python dependencies from requirements.txt
echo "🚀 Installing Python dependencies..."
pip install -r requirements.txt

# Build the documentation using mkdocs
echo "🚀 Building documentation using mkdocs..."
mkdocs build -f nyundocs_v1/mkdocs.yml -d $PUBLISH

# Move into the "publish" directory
cd $PUBLISH

# Config user
git config user.name "Abhrant"
git config user.email "abhiranta@nyunai.com"

# Stage all changes for commit
echo "🚀 Staging changes for commit..."
git add .

# Commit changes with a message
echo "🚀 Committing changes..."
git commit -m "updating docs"

# Push changes to the remote repository's gh-pages branch
echo "🚀 Pushing changes to the remote repository..."
git push origin gh-pages

# Return to the parent directory
cd ..

# Clean up the "publish" directory
echo "🚀 Cleaning up..."
rm -rf $PUBLISH

echo "🚀 Update complete."