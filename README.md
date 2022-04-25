

- rifinire il grafo iterativamente con multiple query
- algoritmi di centralità e communities
- possibilità di cambiare tra mention e mesh_id nella visualizzazione del grafo
- aggiungere pesi agli edges
- Vedere pacchetto pip e caricare su hugginface (o drive) i pesi del modello di biobert REL

Secondario:

- [EFFICIENZA] Per estrarre le relazioni converrebbe chiamare il modello in batch e non una singola frase alla volta

- Guardare l'export in cytoscape mettere roba carina

MARCO:

- Vedere alternative a pymed per scaricare gli articoli (e salvarli in locale)

- Fare un throttler per bern2 (se si vuole scaricare 1000 articoli lui te li scarica a blocchi di 100 / o quant'è il limite)

- In qualche modo pulire la cache

- Con che logica si scaricano gli articoli relativi a una query? In teoria vogliamo quelli più "importanti"?

- Testare il limite del sistema (Quanti articoli possiamo processare in quanto tempo?)


RESOCONTO:

- Usare bern in locale non è fattibile (servono 66GB+ e 32GB di RAM) => Usiamo bern2 online con Cache e Throttler

- Usiamo biobert per estrarre le relazioni in locale (ma non usando una gpu è lento) => Istruzioni per usare su colab

# ProgettoBIO
