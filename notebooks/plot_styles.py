import seaborn as sns

hpi_blue = '#007a9e'
hpi_red = '#b1063a'
hpi_yellow = '#f6a800'

current_palette = "RdBu"

sns.set(font_scale=2.2, rc={'figure.figsize':(20,10),
                            'lines.linewidth': 3})
sns.set_style('white', {'axes.axisbelow': True,
                        'axes.edgecolor': 'black',
                        'axes.facecolor': 'white',
                        'axes.grid': True})

hpi_orange = '#dd6108'
hpi_blue = '#007a9e'
hpi_red = '#b1063a'
hpi_yellow = '#f6a800'
hpi_gray = '#5a6065'

hpi_palette = [hpi_orange, hpi_blue, hpi_red, hpi_yellow, hpi_gray, "#9b59b6"]
sns.set_palette(hpi_palette)
