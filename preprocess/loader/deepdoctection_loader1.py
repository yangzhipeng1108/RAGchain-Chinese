import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt

analyzer = dd.get_dd_analyzer()  # instantiate the built-in analyzer similar to the Hugging Face space demo

df = analyzer.analyze(path = "/path/to/your/doc.pdf")  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)
page = next(doc)

image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
HTML(page.tables[0].html)
print(page.text)