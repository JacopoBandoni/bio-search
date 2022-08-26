# BioSearch

A simple to use literature Mining Python Package

Easy to use literature mining pipeline: automatically query PubMed, download abstracts, perform named entity recognition on chemicals and diseases and generate a knowledge graph using relationship recognition.

The library include network analysis to identify important nodes and edges in the graph and as well functions to easily export to other visualization tools.

The demo.ipynb file include a guide on how to use the library and it demostrates its validity for a set of metabolic diseases, identifying the drugs in the knowledge graph and the links to the associated diseases.

## How to use the no-code interface

Run the server with the following commands:

> export FLASK_APP=server

> python server.py

Then head over to the react client (https://github.com/Pier297/BioSearch)
and execute:

> npm start

## How to use pubmad pyp library

To perform installation run:

> pip install pubmad

Then check out the documentation to look at the available commands (https://pier297.github.io/ProgettoBIO/) or look at demo.ipynb notebook to get an overview of potential use of the commands.

Note: project done in collaboration with William Simoni & Pierpaolo Tarasco. Implemented for the Computational Health Course 2022.
