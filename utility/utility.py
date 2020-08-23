import pandas as pd
import utility.constants as k


def from_csv(data):

    if isinstance(data, str):
        filename, file_ext = os.path.splitext(data)
        if file_ext != ".csv":
            raise FileNotFoundError(f"DataFrame constructor called with incompatible file extension: {file_ext}")

        try:
            csv = pd.read_csv(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"DataFrame constructor called with a file that don't exist: {data} ")

    return pd.DataFrame(csv)


def from_file(filename, sep=",", header='infer', usecols=None, dtype=None):
    return pd.read_csv(filename, sep=sep, header=header, usecols=usecols, dtype=dtype)


def zip_columns(dataframe,elements,newcol = "newcol", dropped = False):

    #prende tutte le colonne di nome elements le accorpa in una nuova colonna con un tupla
    if len(elements)==1:
        dataframe.rename( columns={ list(elements.keys())[0] : newcol} , inplace=True )

    if len(elements) > 1:
        l=[]
        for i in elements:
            l.append(dataframe[i])

        dataframe[newcol] = list(zip(*l))

        if dropped:
           dataframe.drop(elements, axis='columns', inplace=True)


def list_to_string(anytype_list):
    return [str(elem) for elem in anytype_list]


def round_attr(dataframe,column,decimal):
    for elem in dataframe[column]:
        elem = round(elem,decimal)

def date_time_precision(dt, precision):
    result = ""
    if precision == "Year" or precision == "year":
        result += str(dt.year)
    elif precision == "Month" or precision == "month":
        result += str(dt.year) + str(dt.month)
    elif precision == "Day" or precision == "day":
        result += str(dt.year) + str(dt.month) + str(dt.day)
    elif precision == "Hour" or precision == "hour":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month)
    elif precision == "Minute" or precision == "minute":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute)
    elif precision == "Second" or precision == "second":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute) + str(dt.second)
    return result


def frequency_vector(privacy_df, method=k.ELEMENTS_BASED_KNOWLEDGE):

    if method == k.ELEMENTS_BASED_KNOWLEDGE:
        cols= [k.USER_ID,k.ELEMENTS]
    if method == k.SEQUENCE_BASED_KNOWLEDGE or method == k.FULL_SEQUENCE_KNOWLEDGE:
        cols = [k.USER_ID, k.SEQUENCE_ID, k.ELEMENTS]

    frequency = privacy_df.groupby(cols).size().reset_index(name=k.FREQUENCY)
    return frequency.sort_values(by=cols)


def probability_vector(privacy_df, method=k.ELEMENTS_BASED_KNOWLEDGE):
    if method == k.ELEMENTS_BASED_KNOWLEDGE:
        freq = privacy_df.groupby([k.USER_ID, k.ELEMENTS]).size().reset_index(name=k.FREQUENCY)
        dim = privacy_df.groupby([k.USER_ID]).size().reset_index(name=k.TOTAL_FREQUENCY)
        prob = pd.merge(freq, dim, left_on=k.USER_ID, right_on=k.USER_ID)
        prob[k.PROBABILITY] = prob[k.FREQUENCY] / prob[k.TOTAL_FREQUENCY]
        return prob.sort_values( by=[k.USER_ID,k.PROBABILITY])

    if method == k.SEQUENCE_BASED_KNOWLEDGE or method == k.FULL_SEQUENCE_KNOWLEDGE:
        freq = privacy_df.groupby([k.USER_ID, k.SEQUENCE_ID, k.ELEMENTS]).size().reset_index(name=k.FREQUENCY)
        dim = privacy_df.groupby([k.USER_ID, k.SEQUENCE_ID]).size().reset_index(name=k.TOTAL_FREQUENCY)
        prob = pd.merge(freq, dim, left_on=[k.USER_ID,k.SEQUENCE_ID] , right_on=[k.USER_ID,k.SEQUENCE_ID])
        prob[k.PROBABILITY] = prob[k.FREQUENCY] / prob[k.TOTAL_FREQUENCY]
        return prob.sort_values(by=[k.USER_ID,k.SEQUENCE_ID,k.PROBABILITY])

"""def proportion_vector(privacy_df,method = k.ELEMENTS_BASED_KNOWLEDGE):
    if method == k.ELEMENTS_BASED_KNOWLEDGE:
        freq = privacy_df.groupby([k.USER_ID,k.ELEMENTS]).size().reset_index(name=k.FREQUENCY)
        maxf = freq.groupby([k.USER_ID])[k.FREQUENCY].max().reset_index(name=k.MAX_FREQUENCY)
        prop = pd.merge(freq,maxf,left_on=k.USER_ID, right_on=k.USER_ID)
        prop[k.PROPORTION] = prop[k.FREQUENCY] / prop[k.MAX_FREQUENCY]
        return prop.sort_values(by=[k.USER_ID,k.PROPORTION])

    else:
        freq = privacy_df.groupby([k.USER_ID, k.SEQUENCE_ID, k.ELEMENTS]).size().reset_index(name=k.FREQUENCY)
        maxf = freq.groupby([k.USER_ID, k.SEQUENCE_ID])[k.FREQUENCY].max().reset_index(name=k.MAX_FREQUENCY)
        prop = pd.merge(freq, maxf, left_on=[k.USER_ID, k.SEQUENCE_ID], right_on=[k.USER_ID, k.SEQUENCE_ID])
        prop[k.PROPORTION] = prop[k.FREQUENCY] / prop[k.MAX_FREQUENCY]
        return prop.sort_values(by=[k.USER_ID, k.SEQUENCE_ID, k.PROPORTION])"""


def full_list_compare(l1,l2):
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            return False
    return True
