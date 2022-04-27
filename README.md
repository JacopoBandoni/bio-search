# ProgettoBIO

## To run the server

> export FLASK_APP=server

> flask run

Then head over to the react client and execute

> npm start

## OBIETTIVI:
- fixare pymed!
- aggiungere pesi agli edges
- visualizzazione con dash (mettere archi più o meno grossi a seconda del peso, link sugli archi)
- algoritmi di centralità e communities (già integrati in network x)
- migliorare throttler
- Aggiungere biobert QA


Secondario:
- [EFFICIENZA] Per estrarre le relazioni converrebbe chiamare il modello in batch e non una singola frase alla volta
- In qualche modo pulire la cache
- Vedere pacchetto pip e caricare su hugginface (o drive) i pesi del modello di biobert REL
