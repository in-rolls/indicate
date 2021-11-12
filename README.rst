indicate: transliterate indic languages to english
----------------------------------------------------

.. image:: https://ci.appveyor.com/api/projects/status/5wkr850yy3f6sg6a?svg=true
    :target: https://ci.appveyor.com/project/soodoku/indicate
.. image:: https://img.shields.io/pypi/v/indicate.svg
    :target: https://pypi.python.org/pypi/indicate
.. image:: https://pepy.tech/badge/indicate
    :target: https://pepy.tech/project/indicate


Transliterations to/from Indian languages are still generally low quality. One problem is access to data. Another is that there is no standard  transliteration.

For Hindi--English, we build novel dataset for names using the ESPNcricinfo. For instance, see [here](https://www.espncricinfo.com/hindi/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard) for Hindi version of the [English scorecard](https://www.espncricinfo.com/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard). 

We also create a dataset from [election affidavits](https://affidavit.eci.gov.in/CandidateCustomFilter)

We also exploit the [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina).

To overcome the fact that there isn't one standard way of transliteration, we provide k-best transliterations.

Install
----------

We strongly recommend installing `indicate` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install indicate

General API
------------------


Examples
----------


Functions
----------

We expose 6 functions, each of which either take a pandas DataFrame or a
CSV. If the CSV doesn't have a header, we make some assumptions about
where the data is:

- **census\_ln(df, namecol, year=2000)**

  -  What it does:

     - Removes extra space
     - For names in the `census file <ethnicolr/data/census>`__, it appends 
       relevant data of what probability the name provided is of a certain race/ethnicity


 +------------+--------------------------------------------------------------------------------------------------------------------------+
 | Parameters |                                                                                                                          |
 +============+==========================================================================================================================+
 |            | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred             |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name        |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **Year** : *{2000, 2010}, default=2000* year of census to use                                                            |
 +------------+--------------------------------------------------------------------------------------------------------------------------+


-  Output: Appends the following columns to the pandas DataFrame or CSV: 
   pctwhite, pctblack, pctapi, pctaian, pct2prace, pcthispanic. 
   See `here <https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/census/census_2000.pdf>`__ 
   for what the column names mean.

   ::

      >>> import pandas as pd

      >>> from ethnicolr import census_ln, pred_census_ln

      >>> names = [{'name': 'smith'},
      ...         {'name': 'zhang'},
      ...         {'name': 'jackson'}]

      >>> df = pd.DataFrame(names)

      >>> df
            name
      0    smith
      1    zhang
      2  jackson

      >>> census_ln(df, 'name')
            name pctwhite pctblack pctapi pctaian pct2prace pcthispanic
      0    smith    73.35    22.22   0.40    0.85      1.63        1.56
      1    zhang     0.61     0.09  98.16    0.02      0.96        0.16
      2  jackson    41.93    53.02   0.31    1.04      2.18        1.53



Data
----------


Evaluation
------------------------------------------

Authors
----------

Rajasekhar Chintalapati and Gaurav Sood

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
