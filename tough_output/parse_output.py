"""
parse_output.py
-----------------

provides an interface for connecting with
TOUGH2 output files

"""
import pandas as pd
import numpy as np
import os
import re
import pickle

def split_cols(s):
    """ given a header string, split it on white space
    larger than or equal two spaces"""
    return re.split(r'\s{2,}', s)

def iter_file(fname):
    """ just iterate over the file """
    with open(fname) as f:
        for line in f:
            yield line.strip()

HDR_TYPE_MAP = {
        'DELTEX':float,
        "DEN_G":float,
        "DG":float,
        "DL":float,
        "DT":float,
        "DW":float,
        "DX":float,
        "ELEM":str,
        "ELEM.":str,
        "ELEM1":str,
        "ELEM2":str,
        "ELEMENT": str,
        "ENTHALPY":float,
        "FF(GAS)":float,
        "FF(LIQ)":float,
        "FHEAT":float,
        "FLO(BRINE)":float,
        "FLOF":float,
        "FLOH":float,
        "FLOH/FLOF":float,
        "FLOW(LIQ)":float,
        "FLO(GAS)":float,
        "FLO(AQ.)":float,
        "FLO(WTR2)":float,
        "GENERATION RATE":float,
        "INDEX":int,
        "ITERC":int,
        "ITER":int,
        "KCYC":int,
        "KER":int,
        "K(GAS)":float,
        "G(LIQ)":float,
        "KON":int,
        "LOG(K)":float,
        "MAX. RES.":float,
        "MATERIAL":str,
        "NER":int,
        "P":float,
        "PER.MOD":float,
        "PCAP":float,
        "PCAP_GL":float,
        "P(WB)":float,
        "POR":float,
        "POROSITY":float,
        "PRES":float,
        "REL_G":float,
        "REL_L":float,
        "SAT_G":float,
        "SAT_L":float,
        "SG":float,
        "SL":float,
        "SOURCE":str,
        "ST":float,
        "SW":float,
        "T":float,
        "TEMP":float,
        "TOTAL TIME":float,
        "TURB.-COEFF.":float,
        "VAPDIF":float,
        "VEL(AQ.)":float,
        "VEL(GAS)":float,
        "VEL(LIQ)":float,
        "VIS(LIQ)":float,
        "X1":float,
        "X2":float,
        "XAIRG":float,
        "XAIRL":float,
        "XBRINE":float,
        "XBRINEL":float,
        "XRN1":float,
        "XRN1G":float,
        "XRN1L":float,
        "XRN2":float,
        "XRN2G":float,
        "XRN2L":float,
        "XWAT(1)":float,
        "XWAT(2)":float,
        "X_AIR_L":float,
        "X_BRINE_G":float,
        "X_BRINE_L":float,
        "X_WATER_G":float,
        "X_WATER_L":float
        }

def get_types_for_hdr(hdr):
    def gen_keys(val):
        for key in HDR_TYPE_MAP.keys():
            if key in val:
                return HDR_TYPE_MAP[key]
    return [gen_keys[i] for i in hdr]

class OutputFile():
    """ represenets a connection to a TOUGH2 output file, containing
    printouts of various parameters, step by step for each timestep
    """ 
    def __init__(self, fname):
        self.fname = fname
        self.hdr_locs = []
        self.offsets = []
        self._headers = None
        self._data_cols = None
        self._conn_data_cols = None
        self._gener_data_cols = None
        self.read_file()

    def to_pickle(self, outfile):
        with open(outfile, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, infile):
        with open(infile, 'rb') as f:
            return pickle.load(f)

    @property
    def n_times(self):
        l = len(self.hdr_locs)
        if l>0:
            return l
        else:
            self.read_file()
        return self.n_times

    @property
    def times(self):
        return np.array([i["TOTAL TIME"] for i in self.headers])

    @property
    def headers(self):
        """ returns a list of dicts, where each dict is the
        contents of the nth header"""
        if self._headers is None:
            self._headers = [self.get_header(i) 
                    for i in range(self.n_times)]
        return self._headers

    def array_for_keys(self, keys):
        """ given the key of an item in the header, this returns
        a numpy array. Keys can be a list and it will
        return a N*m array"""
        if type(keys) == str:
            keys = [keys]

        return np.array(
                    [[j[i] for i in keys] for j in self.headers]
                    )

    def read_file(self):
        """ read the output file """
        self.hdr_locs = []
        self.offsets = []
        self._headers = None
        self._data_cols = None
        self._conn_data_cols = None
        self._gener_data_cols = None

        with open(self.fname, 'r') as f:
            self._iter_file(f)

    def _iter_file(self, f):
        """ iterate the file line by line and populate hooks to items """

        for ix, line in enumerate(f):
            self.offsets.append(len(line))
            if "TOTAL TIME" in line:
                #hdr = split_cols(line.strip())
                # Locate position of state output headers:
                self.hdr_locs.append(ix)#sum(self.offsets[:ix]))
            if "ELEM." in line and self._data_cols is None:
                self._data_cols = line.strip().split()
                # self._data_cols.append('MATERIAL')  # LC 05/12/2020
            if "ELEM1" in line and self._conn_data_cols is None:
                self._conn_data_cols = line.strip().split()
            if "ELEMENT SOURCE INDEX      GENERATION RATE" in line and self._gener_data_cols is None:
                gener_col_names = line.strip().split()
                for count, gener_col in enumerate(gener_col_names):
                    if gener_col == 'GENERATION':
                        break
                # Combine "GENERATION RATE" into a single entry in col names
                gener_col_names = (gener_col_names[:count] +
                                   [gener_col_names[count] + ' ' + gener_col_names[count + 1]] +
                                   gener_col_names[count + 2:])
                gener_col_names = gener_col_names[:5]
                self._gener_data_cols = gener_col_names


        # print(self.hdr_locs)

    def position_for_line(self, n):
        """ return the position in the file for the line number"""
        return sum(self.offsets[:n])

    def get_header(self, n):
        """ get the nth header"""
        if len(self.hdr_locs) == 0:
            self.read_file()
        with open(self.fname, 'r') as f:
            f.seek(self.position_for_line(self.hdr_locs[n])+self.hdr_locs[n])
            # f.seek(self.position_for_line(self.hdr_locs[n])) # LC 12/05/2020
            l = f.readline()
            names = split_cols(l.strip())
            # names = split_cols(f.readline().strip())
            vals = f.readline().split()
            d = dict()
            for name, val in zip(names, vals):
                try:
                    d[name] = list(
                            map(HDR_TYPE_MAP.get(name, float), [val]))[0]
                except ValueError:
                    d[name] = None
            return d

    '''
    def get_elem_header(self, n):
        """ get the nth element section header """
        with open(self.fname, 'r') as f:
            f.seek(self.position_for_line(self.hdr_locs[n]))
            while True:
                dummy = f.readline()
                if "ELEM." in dummy:
                    return f.readline().strip().split()
    '''

    def n_nodes(self):
        """ returns the number of nodes """
        try:
            return self._nnodes
        except Exception:
            self._nnodes = len(list(get_elem(0)))

    def get_elem(self, n):
        """ get the nth element section as a generator """
        if len(self.hdr_locs) == 0:
            self.read_file()
        with open(self.fname, 'r') as f:
            f.seek(self.position_for_line(self.hdr_locs[n])+self.hdr_locs[n])# LC 12/05/2019
#           f.seek(self.position_for_line(self.hdr_locs[n]))# LC 12/05/2019
            while True:
                dummy = f.readline()
                if "ELEM." in dummy:
                    el_col_names = dummy.strip().split()
                    # el_col_names.append('MATERIAL') # LC 05/12/2020
                    self._data_cols = el_col_names
                    
                    #el_col_names = f.readline().strip().split()
                    break

            #dummy = f.readline() # MH Removed 10/20/20
            while True:
                # Only ignore the 'Carriage return' strings at the beginning and end of line
                l = f.readline().strip('\n')
                if "@" in l:
                    break
                if "tough" in l:
                    break
                if "THE TIME IS" in l:
                    break
                if l.strip()=="":   # MH 17/06/2020
                    continue
                if "ELEM." in l:
                    continue
                if "(" in l:
                    continue
                # Force first 5 characters as first element of line data
                # and split remaining portion of line.
                
#                out = [l[:5]] + l[5:-5].split() + [l[-5:].strip()]
                # out = [l[1:6]] + l[6:-5].split() + [l[-5:].strip()] # LC 12/05/2020
                out = [l[:7].strip()] + l[7:].split()  # MH 17/06/2020
                vals = zip(el_col_names, out)
                
                """
                for nm, val in vals:
                    try:
                        i = HDR_TYPE_MAP.get(nm, float)(val)
                    except Exception as e:
                        print(nm, val)
                        raise e
                """
                for ix, name in enumerate(el_col_names):
                    try:
                        out[ix] = HDR_TYPE_MAP.get(name, float)(out[ix])
                    except ValueError:
                        out[ix] = None
                yield out

    def get_conn(self, n):
        """ get the nth connection section as a generator """
        if len(self.hdr_locs) == 0:
            self.read_file()
        with open(self.fname, 'r') as f:
            f.seek(self.position_for_line(self.hdr_locs[n])+self.hdr_locs[n])
            while True:
                dummy = f.readline()
                if "ELEM1" in dummy:
                    conn_col_names = dummy.strip().split()
                    self._conn_data_cols = conn_col_names
                    break

            dummy = f.readline()
            while True:
                # Only ignore the 'Carriage return' strings at the beginning and end of line
                l = f.readline().strip()

                if "@" in l:
                    # Reached the end of the connection outputs:
                    break
                if l.strip()=="":   # MH 17/-6/2020
                    # Intermediate header information reported in output file:
                    continue
                if "ELEM1" in l:
                    # Intermediate header information (variable name) in output file:
                    continue
                if "(" in l:
                    # Intermediate header information (units) in output file:
                    continue

                # Parse data from connection output string:
                # INDEX length is 7 for tough-mp (> 1e6 connections), 6 for simpler models
                col1 = l.find('.') - l[l.find('.'):0:-1].find(' ')    # Find last space before first floating point val
                # col2 = l[:col1] - l[col1:0:-1].find(' ') # Find last space before col1
                # if col1 - col2 < 10:
                #     # We have isolated only the INDEX
                #     col3 = l[:col2] - l[col2:0:-1].find(' ')  # Find last space before col2
                #     out = ([l[col3:col2].strip()[:5]] +
                #            [l[col3:col2].strip()[-5:]] +
                #            [l[col2:col1].strip()])
                # else:
                #     # We have isolated INDEX and ELEM2:
                #     out = (l[:col2].strip() +
                #            [l[col2:col1].strip()[:5]] +
                #            [l[col2:col1].strip()[5:]])
                # out += l[col1:].split()
                out = ([l[:col1].strip()[:-7].strip()[:-5].strip()] + # Find ELEM1 (stripped string before ELEM2)
                       [l[:col1].strip()[:-7].strip()[-5:].strip()] + # Find ELEM2 (last 5 nonempty digits before index)
                       [l[:col1].strip()[-7:].strip()] +              # Find INDEX (last digits of stripped substring)
                       l[col1:].split())                              # Collect rest of floating-point data with split()
                vals = zip(conn_col_names, out)
                """
                for nm, val in vals:
                    try:
                        i = HDR_TYPE_MAP.get(nm, float)(val)
                    except Exception as e:
                        print(nm, val)
                        raise e
                """
                for ix, name in enumerate(conn_col_names):
                    try:
                        out[ix] = HDR_TYPE_MAP.get(name, float)(out[ix])
                    except ValueError:
                        out[ix] = None
                yield out

    def get_gener(self, n):
        """ get the nth source/sink listing as a generator """
        if len(self.hdr_locs) == 0:
            self.read_file()
        with open(self.fname, 'r') as f:
            f.seek(self.position_for_line(self.hdr_locs[n])+self.hdr_locs[n])
            while True:
                dummy = f.readline()
                if "GENERATION RATE" in dummy:
                    gener_col_names = dummy.strip().split()
                    for count, gener_col in enumerate(gener_col_names):
                        if gener_col == 'GENERATION':
                            break
                    # Combine "GENERATION RATE" into a single entry in col names
                    gener_col_names = (gener_col_names[:count] +
                                       [gener_col_names[count] + ' ' + gener_col_names[count+1]] +
                                       gener_col_names[count+2:])
                    gener_col_names = gener_col_names[:5]
                    self._gener_data_cols = gener_col_names
                    break

            dummy = f.readline()
            while True:
                # Only ignore the 'Carriage return' strings at the beginning and end of line
#                l = f.readline().strip()
                l = f.readline().strip('\n') # LC 12/05/2020
                if "@" in l:
                    # Reached the end of the gener outputs:
                    break
                if l.strip()=="":  # MH 17/06/2020
                    # Intermediate header information reported in output file:
                    continue
                if "GENERATION RATE" in l:
                    # Intermediate header information (variable name) in output file:
                    continue
                if "(" in l:
                    # Intermediate header information (units) in output file:
                    continue
                # Force first 8 characters as first element name,
                # second 5 characters as second element name,
                # and split remaining portion of line.
#                out = [l[:5]] + [l[8:13]] + l[13:].split()
                out = [l[2:7]] + [l[10:16]] + l[16:].split() # LC 12/05/2020
                if len(out) > 5:
                    # This is a heat source/sink
                    continue

                vals = zip(gener_col_names, out)
                """
                for nm, val in vals:
                    try:
                        i = HDR_TYPE_MAP.get(nm, float)(val)
                    except Exception as e:
                        print(nm, val)
                        raise e
                """
                for ix, name in enumerate(gener_col_names):
                    try:
                        out[ix] = HDR_TYPE_MAP.get(name, float)(out[ix])
                    except ValueError:
                        out[ix] = None
                yield out

    def np_array_for_step(self, n):
        """ return a numpy array of node data for the nth step """
        return np.array(list(self.get_elem(n)))

    def dataframe_for_step(self, n):
        """ return the n-th dataframe of all the node data """
        df = pd.DataFrame.from_records(self.get_elem(n)
                , columns=self._data_cols)
        df.set_index(df['ELEM.'], inplace=True)
        return df

    def conn_dataframe_for_step(self, n):
        """ return the n-th dataframe of all the connection data """

        df = pd.DataFrame.from_records(self.get_conn(n), columns=self._conn_data_cols)
        df.set_index(df['INDEX'], inplace=True)
        return df

    def gener_dataframe_for_step(self, n):
        # return the n-th dataframe of all the source/sink data
        df = pd.DataFrame.from_records(self.get_gener(n),
                                       columns = self._gener_data_cols)
        df.set_index(df['ELEMENT'], inplace=True)
        return df

    @property 
    def data_cols(self):
        """ returns a list of the available node data column names 
        
        ordered by col_index for the ith state variable
        """
        return self._data_cols

    @property
    def conn_data_cols(self):
        """ returns a list of the available connection data column names

        ordered by col_index for the ith state variable
        """
        return self._conn_data_cols
    
    def col_index_for_state_variable(self, v):
        """ get the column index for the vth state variable
        (like 'P' ) 

        This is useful for extracting data from np_array_for_step

        """
        try:
            for item in self._data_cols:
                if v == item:
                    return item
        except AttributeError:
            self.get_elem(0)
            return self.col_index_for_state_variable(v)

def time_steps(itr):
    get_line = False
    hdr = None
    for line in itr:
        if get_line:
            if hdr is None:
                raise Exception("Header not found")
            d = dict(zip(hdr, line.split()))
            ts = TimeStep(d, itr)
            get_line = False

        if "TOTAL TIME" in line:
            hdr = split_cols(line)
            get_line = True

"""
def time_steps(fname):
    get_line = False
    with open(fname) as f:
        for line in enumerate(f):
            if get_line:
                get_line=False
                yield(line.strip.split())
                if "TOTAL TIME" in line:
                    get_line = True
"""


def make_sample(fname):
    fout = "temp.txt"
    with open(fout, 'w') as fout:
        with open(fname, 'r') as fin:
            for ix, line in enumerate(fin):
                if ix>305500:
                    break
                fout.write(line)

def sample_file(fname):
    get_next = False
    with open(fname) as f:
        for ix, line in enumerate(f):
            """
            if "tough2" in line:
                print(line.strip())

            if "OUTPUT DATA AFTER" in line:
                print(line.strip())
            if get_next:
                get_next=False
                print(line.strip())
            if "TOTAL TIME" in line:
                get_next =True
            """
            print(line.strip())
            if ix > 20:
                break

if __name__ == "__main__":
    pdir = os.path.join(".", "data_eos5_simu") # LC 12/05/2020
    fname = os.path.join(pdir, "SMA_ZNO_2Dhr_gv1_pv1_gas.out")  # LC 12/05/2020
    # fname = os.path.join(pdir, "mikey.out")
    pckfile = os.path.join(pdir, "temp_output.pck") # LC 12/05/2020
    
#    con = OutputFile(fname)
#    sss=con.dataframe_for_step(0)
    try:
        con = OutputFile.from_pickle(pckfile)
        print("Read from pickle {0}".format(pckfile))
    except Exception as e:
        # raise(e)
        print(fname)
        con = OutputFile(fname)
        con.to_pickle(pckfile)
        print('Pickled the output file')

    t_unique, i_unique = np.unique(con.times, return_index=True)
    # print(i_unique)
    # print(len(con.times))
    # print(len(np.unique(con.times)))
    ind = 196# 59
    # print(con.times[ind]/(365.25*24.0*3600))
    # print(t_unique[ind]/(365.25*24.0*3600))
    print(con.dataframe_for_step(ind)['ELEM.'])
    print(con.conn_dataframe_for_step(ind)['ELEM1'])
    print(con.gener_dataframe_for_step(ind)['ELEMENT'])
