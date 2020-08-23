import pandas as pd
from utility.utility import zip_columns
import utility.constants as k
import pandas.core

class PrivacyDataFrame(pd.DataFrame):
    """ PrivacyDataFrame.

        A PrivacyDataFrame object is a pandas.DataFrame that has five columns user_id, datetime, sequence_id, order,
        elements. PrivacyDataFrame accepts the following keyword arguments:

        Parameters
        ----------
        data : list or dict or pandas DataFrame
            the data that must be embedded into a PrivacyDataFrame.

        user_id : int or str, optional
            the position or the name of the column in `data`containing the user identifier. The default is `k.UID`.

        datetime : int or str, optional
            the position or the name of the column in `data` containing the datetime. The default is `k.DATETIME`.

        sequence_id : int or str, optional
            the position or the name of the column in `data` containing the generic sequence identifier. The default is `k.TID`.

        element_id : int or str, optional
            the position or the name of the column in `data` containing the identifier for preserving the elements order within a sequence.
            The deafult is `k.ORDER`.

        elements : dict, optional
            dictionary which contains as key all names of columns in 'data' i would group in and respective values are related types.
            The default is 'None'.

        timestamp : boolean, optional
            it True, the datetime is a timestamp. The default is `False`.


        Examples
        --------
        >>> # create a PrivacyDataFrame from a retail file
        >>> data = pd.read_csv('../data/_ret_2010_sem1_2000_2.csv', sep=',', header='infer', dtype='str')
        >>> df_ret = PrivacyDataFrame(data, user_id='CustomerID', datetime='InvoiceDate',sequence_id="Invoice", elements={'StockCode':str}, timestamp=True)
        >>> print(df_ret.head())

                  uid            datetime sequence  order elements
        0       12346 2010-01-04 09:24:00   493410      1  TEST001
        2       12346 2010-01-04 09:53:00   493412      1  TEST001
        4892    12346 2010-01-14 13:50:00   494450      1  TEST001
        10177   12346 2010-01-22 13:30:00   495295      1  TEST001
        34056   12346 2010-03-02 13:08:00   499763      1    20682
        ...       ...                 ...      ...    ...      ...
        53556   18283 2010-03-28 13:21:00   502841      13    22507
        53557   18283 2010-03-28 13:21:00   502841      14    21034
        115775  18286 2010-06-24 17:51:00  C513486      1   79323G
        115776  18286 2010-06-24 17:51:00  C513486      2   79323S
        115777  18286 2010-06-24 17:51:00  C513486      3   79323B
    """
    def __init__(self, data, user_id = k.USER_ID, datetime = k.DATETIME, element_id = k.ORDER_ID,
                 sequence_id = k.SEQUENCE_ID, elements = None, timestamp = False):

        d_columns = {user_id : k.USER_ID,
                     datetime : k.DATETIME,
                     sequence_id : k.SEQUENCE_ID,
                     element_id : k.ORDER_ID}

        if isinstance(data, pd.DataFrame):
            df = data.rename(columns=d_columns)

        elif isinstance(data,list):
            df = pd.DataFrame(list(zip(*data))).rename(columns=d_columns)

        elif isinstance(data,dict):
            df = pd.DataFrame(data).rename(columns=d_columns)

        else:
            raise TypeError(f"PrivacyDataFrame constructor called with incompatible data and dtype: {type(data)}")


        # calling superclass constructor
        super().__init__(df)

        # Verifico la consistenza del PrivacDataFrame e dei parametri inseriti
        self._data_check(elements)

        # Controlli sugli attributi che voglio raggruppare in una tupla ed il tipo scelto dall'utente
        if elements != None:

            #Effettuo il cast degli attributi come richiesto dall'utente
            self._data_cast(elements=elements, timestamp=timestamp)

            #Raggruppo gli attributi richiesti dall'utente in una tupla inserita in una colonna del PrivacyDataFrame
            zip_columns(self, elements=elements, newcol=k.ELEMENTS, dropped=True)


    def _data_check(self, elements):

        #COLUMNS CHECK

        #Controllo che gli attributi siano tutti presenti
        if not k.USER_ID in self:
            raise AttributeError(f"PrivacyDataFrame doesn't cointain {k.USER_ID} attribute")

        if not k.SEQUENCE_ID in self:
            raise AttributeError(f"PrivacyDataFrame doesn't cointain {k.SEQUENCE_ID} attribute")

        if k.DATETIME in self and k.ORDER_ID in self:
            sort4elem = self.sort_values(by=[k.USER_ID, k.SEQUENCE_ID, k.ORDER_ID])
            sort4data =self.sort_values(by=[k.USER_ID, k.SEQUENCE_ID, k.ORDER_ID])
            if not sort4elem.equals(sort4data):
                raise AttributeError(f"PrivacyDataFrame {k.DATETIME} attribute doesn't match with {k.ORDER_ID}")

        if k.DATETIME in self and k.ORDER_ID not in self:
            self.sort_values(by=[k.USER_ID, k.SEQUENCE_ID, k.DATETIME], inplace=True)
            self[k.ORDER_ID]=0
            #self._make_order()

        if k.DATETIME not in self and k.ORDER_ID not in self:
            self.sort_values(by=[k.USER_ID,k.SEQUENCE_ID], inplace=True)
            self[k.ORDER_ID] = 0
            self._make_order()

        #ELEMENTS CHECK
        if elements != None:

            # Controllo che tutti gli attributi da raggruppare e da castare siano presenti nel PrivacyDataFrame
            for attr in elements.keys():
                if attr not in self:
                    raise AttributeError(f"PrivacyDataFrame cannot group {elements} in a tuple. Some attributes aren't contained in PrivacyDataFrame")

            # Controllo che tutti gli attributi da raggruppare siano associati ad un tipo
            for type_attr in elements.values():
                if not isinstance(type_attr,type):
                    raise AttributeError(f"PrivacyDataFrame cannot group {elements} in a tuple. Some attributes aren't contained in PrivacyDataFrame")

        #DATETIME deve essere datetime
        if k.DATETIME in self and not pd.core.dtypes.common.is_datetime64_any_dtype(self[k.DATETIME].dtype):
            self[k.DATETIME] = pd.to_datetime(self[k.DATETIME])

    def _make_order(self):
        # Creo l'iteratore
        df_iterator = self.iterrows()

        # Salvo i valori della prima riga
        index, row = next(df_iterator)
        self.loc[index, k.ORDER_ID] = 1
        puid, pseq = row[k.USER_ID], row[k.SEQUENCE_ID]

        # Ordino le sequenze
        val = 2
        for index, row in df_iterator:
            if (row[k.USER_ID] == puid) and (row[k.SEQUENCE_ID] == pseq):
                self.loc[index, k.ORDER_ID] = val
            else:
                val = 1
                self.loc[index, k.ORDER_ID] = 1
                puid, pseq = row[k.USER_ID], row[k.SEQUENCE_ID]
            val += 1


    def _data_cast(self, elements = None, timestamp=False,):

        #Casting timestamp
        if timestamp:
            self[k.DATETIME] = pd.to_datetime(self[k.DATETIME], unit='s')

        #Effettuo il casting delle colonne in base a come richiesto dall'utente
        if elements != None:
            for attr, attr_type in elements.items():
                self[attr] = self[attr].astype(attr_type)

    @staticmethod
    def adjust(privacydf,name=None):
        if k.DATETIME in privacydf:
            cols_order = [k.USER_ID, k.DATETIME, k.SEQUENCE_ID, k.ORDER_ID, k.ELEMENTS]
            cols = [k.USER_ID, k.SEQUENCE_ID, k.DATETIME]
        else:
            cols_order = [k.USER_ID, k.SEQUENCE_ID, k.ORDER_ID, k.ELEMENTS]
            cols = [k.USER_ID, k.SEQUENCE_ID, k.ORDER_ID]

        # ordino il dataframe
        df = privacydf.sort_values(by=cols,inplace=False)

        # ordino le colonne
        df = df[cols_order]

        #elimino righe con almeno un attributo NaN
        df = df.dropna(how='any', axis=0)

        #elimino le righe con elements = POST
        if name!=None:
            df = df[df[k.ELEMENTS]!=name]

        return df


    @property
    def uid(self):
        if k.USER_ID not in self:
            raise AttributeError("The PrivacyDataFrame does not contain the column '%s.'" % k.USER_ID)
        return self[k.USER_ID]

    @property
    def datetime(self):
        if k.DATETIME not in self:
            raise AttributeError("The PrivacyDataFrame does not contain the column '%s.'" % k.DATETIME)
        return self[k.DATETIME]

    @property
    def sequence(self):
        if k.SEQUENCE_ID not in self:
            raise AttributeError("The PrivacyDataFrame does not contain the column '%s.'" % k.SEQUENCE_ID)
        return self[k.SEQUENCE_ID]

    @property
    def args(self):
        if k.ELEMENTS not in self:
            raise AttributeError("The PrivacyDataFrame does not contain the column '%s.'" % k.ELEMENTS)
        return self[k.ELEMENTS]


