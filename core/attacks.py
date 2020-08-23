import pandas as pd
import numpy as np
from abc import abstractmethod
from itertools import combinations
from time import process_time
import utility.constants as k
from core.privacydf import PrivacyDataFrame
from utility.utility import date_time_precision, frequency_vector, probability_vector, full_list_compare


class Attack(object):
    """
    Attack

    Abstract class for a generic attack. Defines a series of functions common to all attacks.
    Provides basic functions to compute risk for all users in a PrivacyDataFrame.
    Requires the implementation of both a matching function and an assessment function, which are attack dependant.

    Parameters
    ----------

    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    """

    def __init__(self, knowledge_length):
        self.knowledge_length = knowledge_length

    @property
    def knowledge_length(self):
        return self._knowledge_length

    @knowledge_length.setter
    def knowledge_length(self, val):
        if val < 1:
            raise ValueError("Il parametro knowledge_length non può essere inferiore ad 1")
        self._knowledge_length = val

    def _all_risks(self, priv_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE,previous_risk=None):
        """
        Computes risk for all the users in the data. It applies the risk function to every individual in the data.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the calculation on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk)

        """
        if uids == None:
            uids = priv_df
        else:
            # ATTENZIONE: se uids è una lista di interi e nel df sono stringhe la isin fallisce
            uids = [str(uid) for uid in uids if type(uid) != str]

            if isinstance(uids, list):
                uids = priv_df[priv_df[k.USER_ID].isin(uids)]

            if isinstance(uids, PrivacyDataFrame) or isinstance(uids, pd.DataFrame):
                uids = priv_df[priv_df[k.USER_ID].isin(uids[k.USER_ID])]

        ##MODIFY
        if previous_risk is not None:
            mask_max_risks = previous_risk[k.PRIVACY_RISK] >= 1
            if mask_max_risks.all() == True:
                risks = previous_risk

            else:
                df_max_risk = previous_risk[mask_max_risks]
                df_not_max_risk = previous_risk[~mask_max_risks]
                uids_not_max_risk = df_not_max_risk[k.USER_ID]
                uids = uids[uids[k.USER_ID].isin(uids_not_max_risk)]
                risks = uids.groupby(k.USER_ID).apply(lambda x: self._privacy_risk(x, priv_df, method=method)).reset_index(
                    name=k.PRIVACY_RISK)
                risks = risks.append(df_max_risk)

        else:
            risks = uids.groupby(k.USER_ID).apply(lambda x: self._privacy_risk(x, priv_df, method=method)).reset_index(
                name=k.PRIVACY_RISK)

        return risks


    def _background_generator(self, single_priv_df, method):
        """
        Return a generator to all the possible background knowledge of length k for a single user_id.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        Yields
        ------
        generator
            a generator to all the possible instances of length k. Instances are tuples with the values of the actual
            records in the combination.
        """
        if method == k.ELEMENTS_BASED_KNOWLEDGE:
            # Genero tutti i casi di un singolo individuo
            size = len(single_priv_df)
            if self.knowledge_length > size:
                cases = combinations(single_priv_df.values, size)
            else:
                cases = combinations(single_priv_df.values, self.knowledge_length)

            return cases

        elif method == k.SEQUENCE_BASED_KNOWLEDGE:
            cases_list = single_priv_df.groupby(k.SEQUENCE_ID).apply(lambda x: self._background_generator(x, k.ELEMENTS_BASED_KNOWLEDGE))
            cases = []
            for cases4seq in cases_list:
                for case4seq in cases4seq:
                    cases.append(case4seq)

            return cases

        elif method == k.FULL_SEQUENCE_KNOWLEDGE:

            # Lista degli indici univoci delle sequenze array([1,2,3,4])
            unique_seqs = single_priv_df[k.SEQUENCE_ID].unique()

            # Tutte i possibili k sottoinsiemi di intere sequenze (1,2)-(1,3)-(1,4)-(2,3)-(2,4)-(3,4)
            size = len(unique_seqs)
            if self.knowledge_length > size:
                cases = combinations(unique_seqs,size)
            else:
                cases = combinations(unique_seqs, self.knowledge_length)

            #inserisco in una lista l'intero contenuto di ogni sequenza
            seqs_values = []
            for case in cases:
                goodseqs = list(case)
                case = single_priv_df[single_priv_df[k.SEQUENCE_ID].isin(goodseqs)]
                seqs_values.append(case)
            return seqs_values

        else:
            raise AttributeError(f"Impossible generate instances: attribute method type unknow.")


    def _privacy_risk(self, single_priv_df, priv_df, method=k.ELEMENTS_BASED_KNOWLEDGE):
        """
        Computes the risk of reidentification of an individual with respect to the entire population in the data.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        Returns
        -------
        float
            the risk for the individual, expressed as a float between 0 and 1
        """
        if method == k.ELEMENTS_BASED_KNOWLEDGE:

            cases = self._background_generator(single_priv_df, k.ELEMENTS_BASED_KNOWLEDGE)
            privacy_risk = 0

            a = process_time()

            for case in cases:

                if self._knowledge_length == 1:
                    elem = (case[-1])[-1]
                    if elem in k.D1:
                        case_risk = k.D1[elem]
                    else:
                        mask = (priv_df[k.ELEMENTS] == elem)
                        uid = (priv_df[mask])[k.USER_ID].unique()
                        case_risk = 1 / len(uid)
                        k.D1[elem] = case_risk
                    if case_risk > privacy_risk:
                        privacy_risk = case_risk
                    if privacy_risk == 1:
                        break

                elif self._knowledge_length == 2:

                    if len(case)==1:
                        case_risk = (case[-1])[-1]

                    else:
                        elem1 = (case[-1])[-1]
                        elem2 = (case[-2])[-1]

                        tup = (elem1,elem2)
                        if tup in k.D2:
                            case_risk = k.D2[tup]
                        else:
                            mask1 = (priv_df[k.ELEMENTS] == elem1)
                            uid1 = (priv_df[mask1])[k.USER_ID].unique()

                            mask2 = (priv_df[k.ELEMENTS] == elem2)
                            uid2 = (priv_df[mask2])[k.USER_ID].unique()

                            uid = list(set(uid1) & set(uid2))

                            case_risk = 1/len(uid)
                            print(case_risk)
                            tup = (elem1,elem2)
                            k.D2[tup]=case_risk

                    if case_risk > privacy_risk:
                        privacy_risk = case_risk
                    if privacy_risk == 1:
                        break

                    """if elem in k.D1:
                        case_risk = k.D1[elem]
                    else:
                        mask = (priv_df[k.ELEMENTS] == elem)
                        uid = (priv_df[mask])[k.USER_ID].unique()
                        case_risk = 1 / len(uid)
                        k.D1[elem] = case_risk"""

                else:
                    case_risk = 1.0 / priv_df.groupby(k.USER_ID).apply(lambda x: self._matching(x, case)).sum()
                    if case_risk > privacy_risk:
                        privacy_risk = case_risk
                    if privacy_risk == 1:
                        break
            k.COUNTER = k.COUNTER + 1
            print(f"Utente: {k.COUNTER} Rischio: {privacy_risk}Tempo:{process_time() - a}")
            return privacy_risk

        elif method == k.SEQUENCE_BASED_KNOWLEDGE:

            cases = self._background_generator(single_priv_df, k.SEQUENCE_BASED_KNOWLEDGE)

            privacy_risk = 0
            for case in cases:
                num = single_priv_df.groupby([k.SEQUENCE_ID]).apply(lambda x: self._matching(x, case)).sum()
                den = priv_df.groupby([k.USER_ID, k.SEQUENCE_ID]).apply(lambda x: self._matching(x, case)).sum()
                case_risk = num / den

                if case_risk > privacy_risk:
                    privacy_risk = case_risk

                if privacy_risk == 1:
                    break

            return privacy_risk

        elif method == k.FULL_SEQUENCE_KNOWLEDGE:
            cases = self._background_generator(single_priv_df, k.FULL_SEQUENCE_KNOWLEDGE)

            privacy_risk = 0
            for case in cases:
                case_risk = 1.0 / priv_df.groupby(k.USER_ID).apply(lambda x: self._full_elements_match(x, case)).sum()

                if case_risk > privacy_risk:
                    privacy_risk = case_risk

                if privacy_risk == 1:
                    break

            return privacy_risk

        else:
            raise AttributeError(f"Impossible compute privacy risk: attribute method type unknow.")

    @abstractmethod
    def evaluate_risk(self, priv_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        pass

    @abstractmethod
    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack. It is used to decide if an instance of background knowledge matches a certain
        sequence. The internal logic of an attack is represented by this function, therefore, it must be implemented
        depending in the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        pass

    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case an entire content of a set of sequence, matches a certain sequence.
        The internal logic of an attack is represented by this function, therefore, it must be implemented depending in
        the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        pass


class ElementsAttack(Attack):
    """
    In a element attack the adversary knows the element which occourred by an individual and matches them against
    elements possessed.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    Example
    -------
    >>> from core import attacks
    >>> from core.privacydf import PrivacyDataFrame
    >>> import utility.constants as k
    >>> from utility.utility import from_file

    >>> # read data
    >>> data_ret = from_file("../data/retail.csv")
    >>> df_ret = PrivacyDataFrame(data_ret,user_id='user',datetime='datetime',sequence_id="seq",
    >>>                           element_id='order', elements={'pname':str})

    >>> # Make an ElementsAttack computing privacy risk with a SEQUENCE_BASED_KNOWLEDGE approach.
    >>> at = ElementsAttack(knowledge_length=2)
    >>> risk = at.evaluate_risk(df_ret,method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
       uid   risk
    0    1  0.500
    1    2  0.500
    2    3  1.000
    3    4  1.000
    4    5  1.000
    5    6  0.125
    6    7  1.000
    7    8  0.500

    >>> # Use the ELEMENT_BASED_KNOWLEDGE method and re-compute the risk.
    >>> risk = at.evaluate_risk(df_ret,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  0.500000
    2    3  1.000000
    3    4  0.200000
    4    5  1.000000
    5    6  0.142857
    6    7  1.000000
    7    8  0.250000

    >>> # Change the knowledge length and assess the risk.
    >>> at.knowledge_length=3
    >>> risk = at.evaluate_risk(df_ret,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  1.000000
    1    2  0.500000
    2    3  1.000000
    3    4  0.200000
    4    5  1.000000
    5    6  0.142857
    6    7  1.000000
    7    8  0.333333

    >>> # Assess the risk related a set of individuals
    >>> risk = at.evaluate_risk(df_ret,method=k.ELEMENTS_BASED_KNOWLEDGE,uids=[4,6,7])
    >>> print(risk)
       uid      risk
    0    4  0.200000
    1    6  0.142857
    2    7  1.000000

    """

    def __init__(self, knowledge_length):
        super(ElementsAttack, self).__init__(knowledge_length)

    def evaluate_risk(self, priv_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        priv_df.sort_values(by=[k.USER_ID, k.DATETIME], ascending=True, inplace=True)
        return self._all_risks(priv_df, uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a element attack, only the elements are used in the matching.
        If a sequence presents the same elements as the ones in the instance, a match is found.
        Multiple elements to the same location are also handled.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        # Creo un df e raggruppo l'istanza da testare
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ = occ.astype(dtype=dict(single_priv_df.dtypes))
        occ = occ.groupby([k.ELEMENTS]).size().reset_index(name=k.COUNT + "case")

        # Raggruppo le informazioni di privacy
        single_priv_grouped = single_priv_df.groupby([k.ELEMENTS]).size().reset_index(name=k.COUNT)

        # Eseguo il merge confrontanto il risultato
        merged = pd.merge(single_priv_grouped, occ, left_on=[k.ELEMENTS], right_on=[k.ELEMENTS])

        if len(merged.index) != len(occ.index):
            return 0

        else:
            condition = merged[k.COUNT] >= merged[k.COUNT + "case"]
            if len(merged[condition].index) != len(occ.index):
                return 0
            else:
                return 1

    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case all the elements of a set of sequence, matches a certain sequence.
        The internal logic of an attack is represented by this function, therefore, it must be implemented depending in
        the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        ordered_single_priv = single_priv_df.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        ordered_case = case.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        for _, case_seq in ordered_case.groupby([k.SEQUENCE_ID]):
            match = ordered_single_priv.groupby([k.SEQUENCE_ID]).apply(
                lambda x: list(x[k.ELEMENTS]) == list(case_seq[k.ELEMENTS]))
            if not match.any():
                return 0
        return 1


class ElementsSequenceAttack(Attack):
    """
    In a elements sequence attack the adversary knows the elements which occourred by an individual and the order in
    which they were visited and matches them against sequences.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    Example
    -------
    >>> from core import attacks
    >>> from core.privacydf import PrivacyDataFrame
    >>> import utility.constants as k
    >>> from utility.utility import from_file

    >>> # Read the data
    >>> data_mob = from_file("../data/mobility.csv")
    >>> df_mob = PrivacyDataFrame(data_mob, user_id='user', datetime='data', sequence_id="sequence",
    >>>                           elements={'lat':float,'long':float}, element_id='order')

    >>> # Make an  ElementsSequenceAttack and compute the privacy risk using SEQUENCE_BASED_KNOWLEDGE approach.
    >>> at = ElementsSequenceAttack(knowledge_length=2)
    >>> risk = at.evaluate_risk(df_mob,method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  1.000000
    2    3  1.000000
    3    4  1.000000
    4    5  0.500000
    5    6  1.000000
    6    7  0.142857
    7    8  0.500000

    >>> # Use an ELEMENT_BASED_KNOWLEDGE method
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid  risk
    0    1   0.5
    1    2   1.0
    2    3   1.0
    3    4   1.0
    4    5   0.5
    5    6   1.0
    6    7   0.2
    7    8   0.5

    >>> # Change the background length and assess the risk
    >>> at.knowledge_length=3
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid  risk
    0    1   1.0
    1    2   1.0
    2    3   1.0
    3    4   1.0
    4    5   0.5
    5    6   1.0
    6    7   0.2
    7    8   1.0

    >>> # Assess the risk related a set of individuals
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE,uids=[1,2,7])
    >>> print(risk)
       uid  risk
    0    1   1.0
    1    2   1.0
    2    7   0.2
    """

    def __init__(self, knowledge_length):
        super(ElementsSequenceAttack, self).__init__(knowledge_length)

    def evaluate_risk(self, priv_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        priv_df = priv_df.reindex(columns=[k.USER_ID, k.DATETIME, k.SEQUENCE_ID, k.ELEMENTS])
        priv_df.sort_values(by=[k.USER_ID, k.DATETIME], ascending=True, inplace=True)
        return self._all_risks(priv_df, uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a elements sequence attack, both the elements and the order of visit are used in the matching.
        If a sequence presents the same elements as the ones in the instance, a match is found.
        Multiple elements to the same location are also handled.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        # Creo un df e l'iteratore a partire dall'istanza, posizionandomi sulla prima riga
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ_iterator = occ.iterrows()
        occ_line = next(occ_iterator)[1]

        count = 0
        for index, row in single_priv_df.iterrows():
            if row[k.ELEMENTS] == occ_line[k.ELEMENTS]:
                count += 1
                try:
                    occ_line = next(occ_iterator)[1]
                except StopIteration:
                    break
        if len(occ.index) == count:
            return 1
        else:
            return 0

    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case all the ordered elements of a set of sequence, matches a certain sequence.
        The internal logic of an attack is represented by this function, therefore, it must be implemented depending in
        the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        ordered_single_priv = single_priv_df.sort_values(by=[k.SEQUENCE_ID])
        ordered_case = case.sort_values(by=[k.SEQUENCE_ID])
        for _ , case_seq in ordered_case.groupby([k.SEQUENCE_ID]):
            match = ordered_single_priv.groupby([k.SEQUENCE_ID]).apply(lambda x: list(x[k.ELEMENTS]) == list(case_seq[k.ELEMENTS]))
            if not match.any():
                return 0
        return 1


class ElementsTimeAttack(Attack):
    """
    In a elements time attack the adversary knows the elements which occur by an individual and the time which they
    were record and matches them against sequences. The precision at which to consider the temporal information can also
    be specified.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.


    time_precision : string, optional
        the precision at which to consider the timestamps for the records.
        The possible precisions are: Year, Month, Day, Hour, Minute, Second. The default is `Hour`

    Example
    -------
    >>> from core import attacks
    >>> from core.privacydf import PrivacyDataFrame
    >>> import utility.constants as k
    >>> from utility.utility import from_file

    >>> # Read data
    >>> data_ret = from_file("../data/retail.csv")
    >>> df_ret = PrivacyDataFrame(data_ret,user_id='user',datetime='datetime',sequence_id="seq",
    >>>                           element_id='order', elements={'pname':str})

    >>> # Make an ElementsTimeAttack and compute privacy risk using SEQUENCE_BASED_KNOWLEDGE approach.
    >>> at = ElementsTimeAttack(knowledge_length=2,precision="Minute")
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid  risk
    0    1   1.0
    1    2   1.0
    2    3   1.0
    3    4   1.0
    4    5   1.0
    5    6   1.0
    6    7   0.5
    7    8   1.0

    >>> # Change the time precision and assess new risk
    >>> at.precision = "Month"
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  0.333333
    2    3  0.500000
    3    4  0.500000
    4    5  0.250000
    5    6  0.500000
    6    7  0.166667
    7    8  0.500000

    >>> # Change background length and assess risk using SEQUENCE_BASED_KNOWLEDGE method
    >>> at.knowledge_length=3
    >>> risk = at.evaluate_risk(df_mob,method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.400000
    1    2  1.000000
    2    3  0.500000
    3    4  1.000000
    4    5  0.333333
    5    6  1.000000
    6    7  0.142857
    7    8  0.333333

    >>> # Assess the risk related a set of individuals
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE,uids=[3,4])
    >>> print(risk)
       uid  risk
    0    3   1.0
    1    4   1.0
    """

    def __init__(self, knowledge_length, precision="Hour"):
        self.precision = precision
        super(ElementsTimeAttack, self).__init__(knowledge_length)

    @property
    def precision(self):
        return self._time_precision

    @precision.setter
    def precision(self, val):
        if val not in k.PRECISION_LEVELS:
            raise ValueError("Possible time precisions are: Year, Month, Day, Hour, Minute, Second")
        self._time_precision = val

    def evaluate_risk(self, privacy_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        if k.DATETIME not in privacy_df:
            raise AttributeError(f"PrivacyDataFrame doesn't contain attribute {k.DATETIME}")

        privacy_df[k.TEMP] = privacy_df[k.DATETIME].apply(lambda x: date_time_precision(x, self.precision))
        privacy_df.sort_values(by=[k.USER_ID, k.DATETIME], ascending=True, inplace=True)
        return self._all_risks(privacy_df, uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a element time attack, both the elements and the time of record are used in matching.
        If a sequence presents the same elements with the same temporal information as in the istance, a match is found.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        # Creo un df e raggruppo l'istanza da testare
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ = occ.groupby([k.ELEMENTS, k.TEMP]).size().reset_index(name=k.COUNT + "case")

        # Raggruppo le informazioni di privacy
        single_priv_grouped = single_priv_df.groupby([k.ELEMENTS, k.TEMP]).size().reset_index(name=k.COUNT)

        # Eseguo il merge confrontando il risultato
        merged = pd.merge(single_priv_grouped, occ, left_on=[k.ELEMENTS, k.TEMP], right_on=[k.ELEMENTS, k.TEMP])

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond = merged[k.COUNT] >= merged[k.COUNT + "case"]
            if len(merged[cond].index) != len(occ.index):
                return 0
            else:
                return 1

    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case all the elements and relatived temporal information of a set of sequence,
        matches a certain sequence. The internal logic of an attack is represented by this function, therefore, it must
        be implemented depending in the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        ordered_single_priv = single_priv_df.sort_values(by=[k.SEQUENCE_ID,k.TEMP,k.ELEMENTS])
        ordered_case = case.sort_values(by=[k.SEQUENCE_ID,k.TEMP,k.ELEMENTS])
        for _ , case_seq in ordered_case.groupby([k.SEQUENCE_ID]):
            match = ordered_single_priv.groupby([k.SEQUENCE_ID]).apply(lambda x: np.array_equal
                                                           (x[[k.TEMP,k.ELEMENTS]], case_seq[[k.TEMP,k.ELEMENTS]]))
            if not match.any():
                return 0
        return 1


class ElementsFrequencyAttack(Attack):
    """
    In a elements frequency attack the adversary knows the elements and the frequency with which the individual occurs
    them, and matches them against frequency vectors. A frequency vector, is an aggregation on elements data showing the
    unique elements occourred by an individual and the frequency with which he records those elements.
    It is possible to specify a tolerance level for the matching of the frequency.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    Example
    -------
    >>> # Read retail data
    >>> data_mob = from_file("./data/retail.csv",dtype=str)
    >>> df_mob = PrivacyDataFrame(data_mob,user_id='user', sequence_id="seq", element_id='order', elements={'pname':str})

    >>> # Make an ElementsFrequencyAttack and assess the risk using ELEMENTS_BASED_KNOWLEDGE approach.
    >>> at = attacks.ElementsFrequencyAttack(knowledge_length=2)
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
      uid      risk
    0   1  1.000000
    1   2  0.500000
    2   3  1.000000
    3   4  0.333333
    4   5  1.000000
    5   6  0.166667
    6   7  1.000000
    7   8  0.333333

    >>> # Change the tolerance and compute risk
    >>> at.error = 0.5
    >>> risk = at.evaluate_risk(df_mob,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
      uid      risk
    0   1  0.500000
    1   2  0.500000
    2   3  1.000000
    3   4  0.200000
    4   5  1.000000
    5   6  0.142857
    6   7  1.000000
    7   8  0.250000

    >>> # Change background length of the adversary and assess the risk with SEQUENCE_BASED_KNOWLEDGE
    >>> at.knowledge_length=3
    >>> risk = at.evaluate_risk(df_mob,method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
      uid   risk
    0   1  0.500
    1   2  0.500
    2   3  1.000
    3   4  1.000
    4   5  1.000
    5   6  0.125
    6   7  1.000
    7   8  0.500

    >>> # Assess the risk related a set of individuals
    >>> risk = at.evaluate_risk(df_mob,method=k.SEQUENCE_BASED_KNOWLEDGE,uids=[6,7,8])
    >>> print(risk)
      uid   risk
    0   6  0.125
    1   7  1.000
    2   8  0.500
    """
    def __init__(self, knowledge_length, error=0.0):
        self.error = error
        super(ElementsFrequencyAttack, self).__init__(knowledge_length)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Margin should be in the interval [0.0,1.0]")
        self._error = val

    def evaluate_risk(self, privacy_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(privacy_df, method=method)
        return self._all_risks(freq, uids=uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a frequency elements attack, both the elements and their frequency are used in the matching.
        If a frequency vector presents the same elements with the same frequency as in the istance, a match is found.
        The tolerance level specified at construction is used to construct and interval of frequency
        and allow for less precise matching.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(columns={k.FREQUENCY: k.FREQUENCY + "case"}, inplace=True)
        merged = pd.merge(single_priv_df, occ, left_on=[k.ELEMENTS], right_on=[k.ELEMENTS])

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond1 = merged[k.FREQUENCY + "case"] >= merged[k.FREQUENCY] - merged[k.FREQUENCY] * self.error
            cond2 = merged[k.FREQUENCY + "case"] <= merged[k.FREQUENCY] + merged[k.FREQUENCY] * self.error
            if len(merged[cond1 & cond2].index) != len(occ.index):
                return 0
            else:
                return 1

    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case all the elements and relatived frequency of a set of sequence, matches a
        certain sequence. The internal logic of an attack is represented by this function, therefore, it must be implemented depending in
        the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        ordered_single_priv = single_priv_df.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        ordered_case = case.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        case_seqs = ordered_case.groupby(k.SEQUENCE_ID)
        for _, case_seq in case_seqs:
            match = ordered_single_priv.groupby(k.SEQUENCE_ID).apply(lambda x: self._lambda_full_freq_match(x,case_seq))
            if not match.any():
                return 0
        return 1


    def _lambda_full_freq_match(self, single_priv, case):
        cond1 = list(case[k.ELEMENTS]) == list(single_priv[k.ELEMENTS])
        cond2,cond3=False,False
        if cond1:
            cond2 = full_list_compare(list(case[k.FREQUENCY]), list(single_priv[k.FREQUENCY] - (single_priv[k.FREQUENCY] * self.error)))
            cond3 = full_list_compare(list(single_priv[k.FREQUENCY] + (single_priv[k.FREQUENCY] * self.error)), list(case[k.FREQUENCY]))
        return cond1 and cond2 and cond3


class ElementsProbabilityAttack(Attack):
    """
    In a elements probability attack the adversary knows the elements of and the probability with which he occurs them,
    and matches them against probability vectors.
    A probability vector, is an aggregation on elements data showing the unique elements records by an individual
    and the probability with which he occurs those locations.
    It is possible to specify a tolerance level for the matching of the probability.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    Example
    -------
    >>> from core import attacks
    >>> from core.privacydf import PrivacyDataFrame
    >>> import utility.constants as k
    >>> from utility.utility import from_file

    >>> # read retail data
    >>> data_ret = from_file("../data/retail.csv")
    >>> df_ret = PrivacyDataFrame(data_ret,user_id='user', sequence_id="seq", element_id='order', elements={'pname':str})

    >>> # Creo un ElementsProbabilityAttack e calcolo il rischio utilizzando il primo approccio.
    >>> at = attacks.ElementsProbabilityAttack(knowledge_length=2)
    >>> risk = at.evaluate_risk(df_ret, method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid  risk
    0    1   1.0
    1    2   1.0
    2    3   1.0
    3    4   1.0
    4    5   1.0
    5    6   1.0
    6    7   1.0
    7    8   1.0

    >>> # Change tolerance and assess risk
    >>> at.error = 1.0
    >>> risk = at.evaluate_risk(df_ret, method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  0.500000
    2    3  1.000000
    3    4  0.500000
    4    5  1.000000
    5    6  0.500000
    6    7  0.500000
    7    8  0.333333

    >>> # Change background length of the adversary and assess the risk with SEQUENCE_BASED_KNOWLEDGE
    >>> at.knowledge_length = 3
    >>> risk = at.evaluate_risk(df_ret, method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  0.500000
    2    3  1.000000
    3    4  1.000000
    4    5  1.000000
    5    6  0.142857
    6    7  1.000000
    7    8  0.500000

    >>> # Assess the risk related a set of individuals
    >>> risk = at.evaluate_risk(df_ret, method=k.SEQUENCE_BASED_KNOWLEDGE,uids=[2,5,6])
    >>> print(risk)
       uid      risk
    0    2  0.500000
    1    5  1.000000
    2    6  0.142857
    """
    def __init__(self, knowledge_length, error=0.0):
        self.error = error
        super(ElementsProbabilityAttack, self).__init__(knowledge_length)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Margin should be in the interval [0.0,1.0]")
        self._error = val

    def evaluate_risk(self, privacy_df, uids=None, method=k.ELEMENTS_BASED_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        prob = probability_vector(privacy_df, method=method)
        return self._all_risks(prob, uids=uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a element frequency attack, both the elements and their frequency are used in the matching.
        If a probability vector presents the same elements with the same probability as in the instancem a match is found.
        The tolerance level specified at construction is used to build and interval of probability and allow
        for less precise matching.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(columns={k.PROBABILITY: k.PROBABILITY + "case"},inplace=True)
        merged = pd.merge(single_priv_df, occ, left_on=[k.ELEMENTS], right_on=[k.ELEMENTS])

        if len(merged.index) != len(occ.index):
            return 0
        else:
            cond1 = merged[k.PROBABILITY + "case"] >= merged[k.PROBABILITY] - merged[k.PROBABILITY] * self.error
            cond2 = merged[k.PROBABILITY + "case"] <= merged[k.PROBABILITY] + merged[k.PROBABILITY] * self.error
            if len(merged[cond1 & cond2].index) != len(merged.index):
                return 0
            else:
                return 1


    def _full_elements_match(self, single_priv_df, case):
        """
        Matching function for the attack with FULL_SEQUENCE_KNOWLEDGE approach. It is used to decide if an instance of
        background knowledge, in this case all the elements and relatived probability of a set of sequence, matches a
        certain sequence. The internal logic of an attack is represented by this function, therefore, it must be implemented depending in
        the kind of the attack.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        ordered_single_priv = single_priv_df.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        ordered_case = case.sort_values(by=[k.SEQUENCE_ID, k.ELEMENTS])
        case_seqs = ordered_case.groupby(k.SEQUENCE_ID)
        for _, case_seq in case_seqs:
            match = ordered_single_priv.groupby(k.SEQUENCE_ID).apply(lambda x: self._lambda_full_prob_match(x, case_seq))
            if not match.any():
                return 0
        return 1


    def _lambda_full_prob_match(self, single_priv, case):
        cond1 = list(case[k.ELEMENTS]) == list(single_priv[k.ELEMENTS])
        cond2,cond3=False,False
        if cond1:
            cond2 = full_list_compare(list(case[k.PROBABILITY]), list(single_priv[k.PROBABILITY] - (single_priv[k.PROBABILITY] * self.error)))
            cond3 = full_list_compare(list(single_priv[k.PROBABILITY] + (single_priv[k.PROBABILITY] * self.error)), list(case[k.PROBABILITY]))
        return cond1 and cond2 and cond3


class ElementsProportionAttack(Attack):
    """
    In a elements proportion attack the adversary knows the coordinates of the unique elements occurred by an individual.
    and the relative proportions between their frequencies of visit, and matches them against frequency vectors.
    A frequency vector is an aggregation on elements data showing the unique elements visited by an individual
    and the frequency with which he visited those locations.
    It is possible to specify a tolerance level for the matching of the proportion.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    Example
    -------
    >>> from core import attacks
    >>> from core.privacydf import PrivacyDataFrame
    >>> import utility.constants as k
    >>> from utility.utility import from_file

    >>> # Read retail data
    >>> data = from_file("../data/retail.csv")
    >>> priv = PrivacyDataFrame(data,user_id='user', sequence_id="seq", element_id='order', elements={'pname':str})

    >>> # Make an ElementsProportionAttack and assess the risk using ELEMENTS_BASED_KNOWLEDGE approach.
    >>> at = attacks.ElementsProportionAttack(knowledge_length=2)
    >>> risk = at.evaluate_risk(priv, method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  1.000000
    1    2  0.500000
    2    3  1.000000
    3    4  0.333333
    4    5  1.000000
    5    6  0.166667
    6    7  1.000000
    7    8  0.333333

    >>> # Change the tolerance and compute privacy risk.
    >>> at.error=0.8
    >>> risk=at.evaluate_risk(priv,method=k.ELEMENTS_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  1.000000
    1    2  0.500000
    2    3  0.333333
    3    4  0.333333
    4    5  1.000000
    5    6  0.166667
    6    7  0.500000
    7    8  0.333333

    >>> # Change background knowledge and assess the risk using SEQUENCE_BASED_KNOWLEDGE method.
    >>> at.knowledge_length=3
    >>> risk=at.evaluate_risk(priv,method=k.SEQUENCE_BASED_KNOWLEDGE)
    >>> print(risk)
       uid      risk
    0    1  0.500000
    1    2  0.500000
    2    3  1.000000
    3    4  1.000000
    4    5  1.000000
    5    6  0.142857
    6    7  1.000000
    7    8  0.500000

    >>> # Assess the risk related a set of individuals
    >>> risk=at.evaluate_risk(priv,method=k.SEQUENCE_BASED_KNOWLEDGE,uids=[2,6,8])
    >>> print(risk)
       uid      risk
    0    2  0.500000
    1    6  0.142857
    2    8  0.500000
    """
    def __init__(self, knowledge_length, error=0.0):
        self.error = error
        super(ElementsProportionAttack, self).__init__(knowledge_length)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Margin should be in the interval [0.0,1.0]")
        self._error = val

    def evaluate_risk(self, privacy_df, uids=None, method=k.FULL_SEQUENCE_KNOWLEDGE, previous_risk=None):
        """
        Abstract function to assess privacy risk for a PrivacyDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        priv_df: PrivacyDataFrame
            the dataframe against which to calculate risk.

        uids : PrivacyDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the privacy data. Default values is None
            in which case risk is computed on all users in priv_df. The default is `None`.

        method : int, optional
            the approach which compute privacy risk. It can be calculated in three different methods. Default approach is
            ELEMENTS_BASED_KNOWLEDGE, in that case it consider all records of a specified user.

        previous_risk : PrivacyDataFrame, optional
            the previous DataFrame with the privacy risk for each user. It used to improve the efficenty during the
            risk computation, in particular it calculate privacy risk for each user whose previous risk is less than
            one. Default value is 'None', in which case risk is computed for each user.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(privacy_df, method=method)
        return self._all_risks(freq, uids=uids, method=method, previous_risk=previous_risk)

    def _matching(self, single_priv_df, case):
        """
        Matching function for the attack.
        For a proportion elements attack, both the elements and their relatived proportion of frequency are used in the
        matching. The proportion are calculated with respect to the most frequent element found in the instance.
        If a frequency vector presents the same elements with the same proportions of frequency of
        visit as in the instance, a match is found.
        The tolerance level specified at construction is used to build an interval of proportion
        and allow for less precise matching.

        Parameters
        ----------
        single_priv_df : PrivacyDataFrame
            the privacy dataframe related to a single individual.

        case : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the sequence, 0 otherwise.
        """
        occ = pd.DataFrame(data=case, columns=single_priv_df.columns)
        occ.rename(columns={k.FREQUENCY: k.FREQUENCY + "case"}, inplace=True)
        merged = pd.merge(single_priv_df, occ, left_on=[k.ELEMENTS], right_on=[k.ELEMENTS])

        if len(merged.index) != len(occ.index):
            return 0
        else:
            merged[k.PROPORTION + "case"] = merged[k.FREQUENCY + "case"] / merged[k.FREQUENCY + "case"].max()
            merged[k.PROPORTION] = merged[k.FREQUENCY] / merged[k.FREQUENCY].max()
            cond1 = merged[k.PROPORTION + "case"] >= merged[k.PROPORTION] - (merged[k.PROPORTION] * self.error)
            cond2 = merged[k.PROPORTION + "case"] <= merged[k.PROPORTION] + (merged[k.PROPORTION] * self.error)
            if len(merged[cond1 & cond2].index) != len(occ.index):
                return 0
            else:
                return 1