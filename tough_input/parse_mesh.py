"""
Parse a TOUGH2 MESH file

For python 3.4
"""

import os
import pickle
import numpy as np
import pandas as pd

def float_or_none(s):
    """ converts a string to a float; if it is empty, it will return None"""
    try:
        return float(s)
    except ValueError:
        return None

class Eleme():
    """ represents an Eleme record.  See p172 of the TOUGH2 manual for what the
    properties mean.  """
    def __init__(self, name, nseq, nadd, ma1, ma2, volx, ahtx, pmx, x, y, z):
        self.name = name
        self.nseq = nseq
        self.nadd = nadd
        self.ma1 = ma1
        self.ma2 = ma2
        self.volx = volx
        self.ahtx = ahtx
        self.pmx = pmx
        self.x = x
        self.y = y
        self.z = z
        self.connections = []         # List of connection indices of which this element is a part
        self.is_n1 = []               # List of booleans indicating if Eleme is n1 in each of self.connections
        self.connected_elements = []  # List of connected elements

    def as_numpy_array(self):
        # Generate numpy arrays of ELEME
        dt = np.dtype([('name', 'U5'),
                       ('nseq', 'U5'),
                       ('nadd', 'U5'),
                       ('ma1', 'U3'),
                       ('ma2', 'U2'),
                       ('ma', 'U5'),
                       ('volx', np.float64),
                       ('ahtx', np.float64),
                       ('pmx', np.float64),
                       ('x', np.float64),
                       ('y', np.float64),
                       ('z', np.float64), ])

        data_eleme = np.empty(1, dtype=dt)

        data_eleme['name'] = self.name
        data_eleme['nseq'] = self.nseq
        data_eleme['nadd'] = self.nadd
        data_eleme['ma1'] = self.ma1
        data_eleme['ma2'] = self.ma2
        data_eleme['ma'] = self.ma1 + self.ma2
        data_eleme['volx'] = self.volx
        data_eleme['pmx'] = 1.0 if self.pmx is None else self.pmx
        data_eleme['ahtx'] = 0.0 if self.ahtx is None else self.ahtx
        data_eleme['x'] = self.x
        data_eleme['y'] = self.y
        data_eleme['z'] = self.z

        return data_eleme


class ElemeCollection():
    """ represents an ordered set of Elements, as read from the mesh file """
    def __init__(self, fname):
        self.fname = fname
        self.elements = []
        self.name2idx = dict()

    def proc_nodes(self, ):
        for idx, node in enumerate(gen_nodes(self.fname)):
            self.elements.append(node)
            self.name2idx[node.name] = idx

    def __getitem__(self, item):
        if type(item) == int:
            # item is element listing index
            return self.elements[item]
        elif type(item) == str:
            # item is an element name
            return self.elements[self.name2idx[item]]
        elif type(item) == list:
            return_list = []
            for i_item in item:
                return_list.append(self[i_item])
            return return_list

    def __len__(self):
        return len(self.elements)

    def as_numpy_array(self):
        # Generate numpy arrays of ELEME
        dt = np.dtype([('name', 'U5'),
                       ('nseq', 'U5'),
                       ('nadd', 'U5'),
                       ('ma1', 'U3'),
                       ('ma2', 'U2'),
                       ('ma', 'U5'),
                       ('volx', np.float64),
                       ('ahtx', np.float64),
                       ('pmx', np.float64),
                       ('x', np.float64),
                       ('y', np.float64),
                       ('z', np.float64), ])

        data_eleme = np.empty(len(self.elements), dtype=dt)

        for i_el, elem in enumerate(self.elements):
            # data_eleme[i_el] = elem.as_numpy_array()
            data_eleme['name'][i_el] = elem.name
            data_eleme['nseq'][i_el] = elem.nseq
            data_eleme['nadd'][i_el] = elem.nadd
            data_eleme['ma1'][i_el] = elem.ma1
            data_eleme['ma2'][i_el] = elem.ma2
            data_eleme['volx'][i_el] = elem.volx
            data_eleme['pmx'][i_el] = 1.0 if elem.pmx is None else elem.pmx
            data_eleme['ahtx'][i_el] = 0.0 if elem.ahtx is None else elem.ahtx
            data_eleme['x'][i_el] = elem.x
            data_eleme['y'][i_el] = elem.y
            data_eleme['z'][i_el] = elem.z
            data_eleme['ma'][i_el] = (elem.ma1 + elem.ma2).strip()

        return data_eleme

    def update_from_numpy_array(self, data_eleme, col_names = None, idx = None):

        if idx is None:
            # Update all elements:
            idx = np.arange(len(self.elements))
        if col_names is None:
            # Update all columns
            col_names = ['name', 'nseq', 'nadd', 'ma1', 'ma2', 'volx', 'pmx', 'ahtx', 'x', 'y', 'z']

        if len(data_eleme) != len(self):
            # At least one element has been added or removed, so regenerate new list of Eleme objects
            print('The length of the numpy array data_eleme does not match the number of elements in the mesh'
                  'object.')
            exit()

        else:
            # No elements were added or removed.  Only the attributes listed in col_names have been changed
            for i_el in idx:
                for col_name in col_names:
                    # Update only columns provided in col_names
                    setattr(self.elements[i_el], col_name, data_eleme[col_name][i_el])

        return None

    def change_ma_of_elements(self, elem_list, to_ma, eleme_data=None):

        if eleme_data is None:
            # Changes materials type of all elements whose indices are provided in elem_list to material type to_ma
            eleme_data = self.as_numpy_array()

        eleme_data['ma1'][elem_list] = to_ma[:-2]
        eleme_data['ma2'][elem_list] = to_ma[-2:]
        self.update_from_numpy_array(eleme_data, col_names=['ma1', 'ma2'], idx=elem_list)

        return None

    def to_file(self, f):
        # Print ELEME list to file 'f' (either filename or handle)
        hdr = 'ELEME----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        data_eleme = self.as_numpy_array()
        names = list(data_eleme.dtype.names)
        new_names = names[:5] + names[6:]
        fmt = ['%5s', '%5s', '%5s', '%3s', '%2s', '%10.4E', '%10.4E', '%10.4E', '%10.4E', '%10.4E', '%10.4E']
        np.savetxt(f, data_eleme[new_names], header=hdr, delimiter='', fmt=fmt, comments='')

        return None

    def displace(self, delta_x, delta_y, delta_z):

        for elem in self.elements:
            elem.x += delta_x
            elem.y += delta_y
            elem.z += delta_z

        return None

'''
class Incon():

    def __init__(self, element, X, nseq=None, nadd=None, porx=None):
        self.element = element
        self.X = X
        self.nseq = nseq
        self.nadd = nadd
        self.porx = porx

    @property
    def num_X(self):
        return len(self.X)

    def to_file(self, f, append=True):

        incon_dat = '{:>5}'.format(self.element)
        incon_dat += ' '*5 if self.nseq is None else '{:>5}'.format(self.nseq)
        incon_dat += ' '*5 if self.nadd is None else '{:>5}'.format(self.nadd)
        incon_dat += ' ' *15 if self.porx is None else '{:>15.9E}'.format(self.porx)
        incon_dat += '\n'

        for x in self.X:
            incon_dat += ' {:>19.13E}'.format(x)

        if not append:
            f = open(f,'w')

        f.write(incon_dat)

        if not append:
            f.close()

        return None


class InconCollection():

    def __init__(self):
        self.incons = []
        self.footer = None

    def append(self, incon):
        self.incons.append(incon)
        return None

    def to_file(self, fname):

        f = open(fname, 'w')
        hdr_dat = 'INCON----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        f.write(hdr_dat)
        for incon in self.incons:
            f.write('\n')
            incon.to_file(f)
        f.write('\n')

        if self.footer is not None:
            f.write('\n')
            f.write(self.footer)
            f.write('\n')

        f.close()

        return None


class Gener():

    """ represents an entry into the GENER block, with source/sink terms """
    def __init__(self, element, code_name, gen_type, gx, nseq=None, nadd=None, nads=None, t_gen=None, ex=None, hg=None):
        self.element = element
        self.code_name = code_name
        self.nseq = nseq
        self.nadd = nadd
        self.nads = nads
        self.type = gen_type
        self.t_gen = t_gen
        self.gx = gx
        self.ex = ex
        self.hg = hg

    @property
    def ltab(self):
        if type(self.gx) is float:
            return 1
        else:
            return len(self.gx)

    @property
    def itab(self):
        if self.ex is None:
            return 0
        elif type(self.ex) is float:
            return 1
        else:
            return len(self.ex)

    def to_file(self, f, append=True):

        hdr_dat = '{:>5}{:>5}'.format(self.element, self.code_name)
        hdr_dat += ' '*5 if self.nseq is None else '{:>5}'.format(self.nseq)
        hdr_dat += ' '*5 if self.nadd is None else '{:>5}'.format(self.nadd)
        hdr_dat += ' '*5 if self.nads is None else '{:>5}'.format(self.nads)
        hdr_dat += '{:>5}'.format(self.ltab)
        hdr_dat += ' '*5
        hdr_dat += '{:>4}'.format(self.type)
        hdr_dat += 'E' if self.itab > 1 else ' '
        if self.ltab > 1:
            hdr_dat += ' '*10
        else:
            hdr_dat += '{:>10.4E}'.format(self.gx) if self.gx >= 0.0 else '{: 9.3E}'.format(self.gx)
        hdr_dat += ' '*10 if (self.itab > 1 or self.ex is None) else '{:>10.4E}'.format(self.ex)
        hdr_dat += ' '*10 if self.hg is None else '{:>10.4E}'.format(self.hg)

        if not append:
            # Writing a new file, in which case 'f' is a file handle
            f = open(f, 'w')
        f.write(hdr_dat)

        if self.ltab > 1:
            n_lines = int(np.ceil(self.ltab/4))
            tab_data = [self.t_gen, self.gx, self.ex] if self.itab > 1 else [self.t_gen, self.gx]
            for dat in tab_data:
                n_written = 0
                for _ in np.arange(n_lines):
                    f.write('\n')
                    line_dat = ''
                    n_write = np.min([self.ltab-n_written, 4])
                    for entry in np.arange(n_write):
                        line_dat += '{:>14.7E}'.format(dat[n_written + entry])
                    f.write(line_dat)
                    n_written += n_write

        return None


class GenerCollection():
    """Represents an entire GENER block"""
    def __init__(self, mop12, ):
        self.mop12 = mop12
        self.geners = []

    def __len__(self):
        return len(self.geners)

    def __getitem__(self, item):
        return self.geners[item]

    def append(self, gener):
        self.geners.append(gener)
        return

    def to_file(self, fname):

        hdr_dat = 'GENER----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        f = open(fname, 'w')
        f.write(hdr_dat)

        for gener in self.geners:
            f.write('\n')
            gener.to_file(f)

        f.write('\n')
        f.close()

        return None

'''

class Mesh():
    """ represents a mesh; nodes and their connections """
    def __init__(self, fname):
        self.nodes = ElemeCollection(fname)
        self.nodes.proc_nodes()
        self.connections = ConneCollection(fname)
        self.connections.proc_conne()
        self.proc_nodes()
        self._points = None

    def connected_elements(self, node):
        """ given a node name (str), return
        the connected element index """

        con_els = []  # Initializing list of connected elements
        i_con_els = []
        is_n1 = []

        for icon in self.connections.connections_for_node[node]:
            con = self.connections[icon]
            n1 = con.name1
            n2 = con.name2
            if ((node != n1)): # and (self.nodes.name2idx[n1] not in con_els)): MH 10/20/20
                # node is n2
                is_n1.append(False)
                con_els.append(self.nodes.name2idx[n1])
                i_con_els.append(self.nodes.name2idx[n1])
            elif ((node != n2)): # and (self.nodes.name2idx[n2] not in con_els)): MH 10/20/20
                # note is n1
                is_n1.append(True)
                con_els.append(self.nodes.name2idx[n2])
                i_con_els.append(self.nodes.name2idx[n2])

        isrt = np.argsort(i_con_els) # Arranges lists in order that connected nodes appear in ELEME list

        return np.array(con_els)[isrt].tolist(), np.array(is_n1)[isrt].tolist(), isrt

    def proc_nodes(self):
        # The Mesh 'proc_nodes' routine populates the connected_elements list for each Eleme object
        for elem in self.nodes.elements:
            elem.connected_elements, elem.is_n1, isrt = self.connected_elements(elem.name)
            elem.connections = np.array(self.connections.connections_for_node[elem.name])[isrt].tolist()

    @property
    def points(self):
        """ return a numpy array of [x, y, z, vol, ma1, ma2]
        points ordered by node order """
        if self._points is None:
            n = len(self.nodes)
            self._points = np.zeros([n,4])
            for ix, n in enumerate(self.nodes):
                self._points[ix,:] = [n.x, n.y, n.z, n.volx]#, n.ma1, n.ma2]
            return self.points
        return self._points

    def as_data_frame(self):
        """ return a pandas dataframe.  Index is the node name,
        has columns x, y, z, and volx"""
        #data = self.points
        names = [i.name for i in self.nodes]

        df = pd.DataFrame.from_records(
            data=([n.x, n.y, n.z, n.volx, n.ma1, n.ma2] for n in self.nodes),
            index=names, columns=["x","y","z","volx", "ma1","ma2"])

        df['ma'] = df['ma1']+df['ma2']
        keys = df['ma'].unique()
        z = lambda x: float(np.where(keys == x)[0][0])
        df['ma_code'] = df['ma'].map(z)
        return df

    def remove_nodes_of_type(self, ma):
        # This routine finds all elements with materials type 'ma' (a string), removes from the ELEME block,
        # and removes their instances in the CONNE block.

        tmp_els = []
        tmp_cons = []
        for elem in self.nodes.elements:
            if elem.ma1 + elem.ma2 != ma:
                tmp_els.append(elem)


        for conn in self.connections.connections:

            ma_1 = (self.nodes.elements[self.nodes.name2idx[conn.name1]].ma1 +
                    self.nodes.elements[self.nodes.name2idx[conn.name1]].ma2)
            ma_2 = (self.nodes.elements[self.nodes.name2idx[conn.name2]].ma1 +
                    self.nodes.elements[self.nodes.name2idx[conn.name2]].ma2)
            if ma_1 != ma and ma_2 != ma:
                tmp_cons.append(conn)

        self.nodes.elements = tmp_els
        # self.nodes.proc_nodes()
        self.connections.connection = tmp_cons
        # self.connections.proc_conne()
        # self.proc_nodes()

        return None

    def replace_nodes_of_type(self, from_ma, to_ma, bound_el=False, bound_el_name=None, new_vol=None, d12=None,
                              new_ahtx=0.0, new_pmx=1.0, new_x=0.0, new_y=0.0, new_z=0.0,
                              elem_data=None, iex_rm=None, new_els=None, update_elements=True,
                              conn_data=None, icx_rm=None, update_connections=True):

        # Changes material type of all nodes having type "from_ma" to "to_ma".  If bound_el flag is set,
        # all of these elements are replaced by a single boundary element having name provided by the user
        # (bound_el_name), which must follow the TOUGH convention for element names (3 character string followed
        # by two integers).  The volume of the element will be changed to variable new_vol if provided by the user.
        # Otherwise, it will be the cumulative volume of all elements being replaced.  Likewise, the node distances
        # d1 and d2 in the connections list will change to variable d12 if provided by the user.  Otherwise, these
        # values will remain unchanged.

        if elem_data is None:
            # Generate a fresh numpy array of Eleme data:
            elem_data = self.nodes.as_numpy_array()

        # elem_list = np.where(np.logical_and(elem_data['ma1'] == from_ma[:-2], elem_data['ma2'] == from_ma[-2:]))[0]
        elem_list = np.nonzero(elem_data['ma'] == from_ma)[0]
        el_list = elem_data['name'][elem_list]

        if (from_ma != to_ma) and (not bound_el):
            # This is a simple MA type swap. No extra operations are necessary.
            elem_data['ma1'][elem_list] = to_ma[:-2]
            elem_data['ma2'][elem_list] = to_ma[-2:]
            if update_elements:
                self.nodes.update_from_numpy_array(elem_data, col_names=['ma1', 'ma2'], idx=elem_list)
            return elem_data

        if bound_el:
            if iex_rm is None:
                iex_rm = elem_list
            else:
                iex_rm = np.unique(np.append(iex_rm, elem_list))
            # All the elements of material type to_ma will be replaced with a single boundary element
            volx = np.sum(elem_data['volx'][elem_list])
            if update_elements:
                # Delete elements provided in iex_rm index list from list of Eleme objects and numpy array
                self.nodes.elements = np.delete(np.array(self.nodes.elements), iex_rm).tolist()
                elem_data = np.delete(elem_data, iex_rm)
            # Generate a new boundary element (Eleme object):
            new_el = Eleme(bound_el_name, '', '', to_ma[:-2], to_ma[-2:], volx if new_vol == None else new_vol,
                           new_ahtx, new_pmx, new_x, new_y, new_z)
            elem_data = np.append(elem_data, new_el.as_numpy_array()[0])
            if new_els is None:
                # Start list of new Eleme objects to append to Eleme object list
                new_els = [new_el]
            else:
                # Add to list of new Eleme objects to append to Eleme object list
                new_els.append(new_el)
            # Add new element object to Mesh:
            if update_elements:
                # Add new Eleme objects to Eleme object list:
                self.nodes.elements += new_els
            if conn_data is None:
                # Generate a fresh numpy array of ConneCollection data:
                conn_data = self.connections.as_numpy_array()

            # Find connections having n1 as a boundary element:
            idx_1 = np.nonzero(np.isin(conn_data['name1'], el_list))[0]
            # Find connections having n2 as a boundary element:
            idx_2 = np.nonzero(np.isin(conn_data['name2'], el_list))[0]
            # Find connections having both n1 and n2 as a boundary element
            idx_12, i_rm_1, i_rm_2 = np.intersect1d(idx_1, idx_2, assume_unique=True, return_indices=True)

            if icx_rm is None:
                # First pass
                icx_rm = idx_12.tolist()
            else:
                # This routine has been run before, so make sure all elements that have been higlighted for removal
                # are considered.
                # icx_rm_new = np.intersect1d(np.nonzero(np.isin(conn_data['name1'], elem_data['name'][iex_rm]))[0],
                #                             np.nonzero(np.isin(conn_data['name2'], elem_data['name'][iex_rm]))[0]).tolist()
                # icx_rm = np.unique(np.append(icx_rm, icx_rm_new)).tolist()
                icx_rm = np.unique(np.append(icx_rm, idx_12)).tolist()

            # Then remove those indices from the list of connections to be modified
            idx_1 = np.delete(idx_1, i_rm_1)
            idx_2 = np.delete(idx_2, i_rm_2)

            # Modify element names from entries of CONNE list:
            conn_data['name1'][idx_1] = bound_el_name
            conn_data['name2'][idx_2] = bound_el_name

            # Update attributes of Connection objects:
            if d12 is None:
                # Only update element name in CONNE entry:
                self.connections.update_from_numpy_array(conn_data, col_names=['name1'], idx=idx_1)
                self.connections.update_from_numpy_array(conn_data, col_names=['name2'], idx=idx_2)
            else:
                # Update both element name and d1/d2 in CONNE entry:
                conn_data['d1'][idx_1] = d12
                conn_data['d2'][idx_2] = d12
                # Find instances where both connections have d1 = d2 = d12
                # (connected to a boundary element from a previous pass):
                idx_multi_bounds = np.where((conn_data['d1'] == d12) & (conn_data['d2'] == d12))[0]
                if idx_multi_bounds.tolist():
                    # Remove connections to boundary elements created from previous pass (assuming the same d12 was
                    # used in that previous pass):
                    icx_rm = np.unique(np.append(icx_rm, idx_multi_bounds)).tolist()
                self.connections.update_from_numpy_array(conn_data, col_names=['name1', 'd1'], idx=idx_1)
                self.connections.update_from_numpy_array(conn_data, col_names=['name2', 'd2'], idx=idx_2)

            # Remove connection entries connecting two boundary elements from Conne object list:
            if update_connections:
                # Connections with two boundary elements will be removed from the conn_data numpy array
                # and ConneCollection object:
                conn_data = np.delete(conn_data, icx_rm)
                self.connections.connections = np.delete(np.array(self.connections.connections), icx_rm).tolist()
                # Reset list of connections to remove:
                icx_rm = None

        return elem_data, new_els, iex_rm, conn_data, icx_rm

    def to_file(self, fname='MESH'):

        # Writes Mesh object as ELEME and CONNE blocks to file fname

        f = open(fname, 'w')
        self.nodes.to_file(f)
        f.write('\n')
        self.connections.to_file(f)
        f.write('\n')
        f.close()

        return None

    @classmethod
    def from_pickle(cls, file):
        """ return a mesh from a pickled file """
        with open(file, "rb") as f:
            return pickle.load(f)

    def to_pickle(self, file):
        """ dump to pickle """
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def gen_nodes(fname):
    """ iterate through f and yield a node for each item """
    with open(fname, 'r') as f:
        idx = None
        for line in f:
            if "ELEME" in line:
                idx = 0
                continue
            if idx is not None:
                if line.strip() == "":
                    break
                name = line[0:5]
                """ five character code name of element"""
                nseq = line[5:10]
                """ number of additional elements having the same volume """
                nadd = line[10:15]
                """ increment between hte code numbers of two successsive elements """
                ma1 = line[15:18]
                """reserved code prefix """
                ma2 = line[18:20]
                """ reserved code suffux"""
                volx = float_or_none(line[20:30])
                """ element volume [m3] """
                ahtx = float_or_none(line[30:40])
                """ interface area [m2] """
                pmx = float_or_none(line[40:50])
                """ permeability modifier """
                x = float_or_none(line[50:60])
                y = float_or_none(line[60:70])
                z = float_or_none(line[70:80])
                """ cartesian coordinates of grid block centers.  """
                yield Eleme(name, nseq, nadd, ma1, ma2, volx, ahtx, pmx, x, y, z)
                idx+=1

class Conne():
    """ introduces information for the connections (interfaces) between elements
    see appendix E, p173 of the TOUGH2 manual """

    def __init__(self, name1, name2, nseq, nad1, nad2, isot, d1, d2, areax, betax, sigx):
        self.name1 = name1
        self.name2 = name2
        self.nseq = nseq
        self.nad1 = nad1
        self.nad2 = nad2
        self.isot = isot
        self.d1 = d1
        self.d2 = d2
        self.areax = areax
        self.betax = betax
        self.sigx = sigx

class ConneCollection():
    """ interface to a collection of connection data """
    def __init__(self, filename):
        self.filename = filename
        self.connections_for_node = dict()
        self.connections = []

    def proc_conne(self):
        """ iterate the connections file and get """
        for idx, conn in enumerate(gen_connections(self.filename)):
            self.connections.append(conn)
            n1 = conn.name1
            n2 = conn.name2
            for node in [n1, n2]:
                try:
                    if idx not in self.connections_for_node[node]:
                        self.connections_for_node[node].append(idx)
                except Exception:
                    self.connections_for_node[node] = [idx]

    def __getitem__(self, item):
        if type(item) == int:
            return self.connections[item]
        if type(item) == str:
            return [self.connections[i] for i in self.connections_for_node[item]]
        if type(item) == Eleme:
            return [self.connections[i] for i in self.connections_for_node[item.name]]

    def __len__(self):
        return len(self.connections)

    def as_numpy_array(self):
        # Generate numpy array of CONNE data
        dt = np.dtype([('name1', 'U5'),
                       ('name2', 'U5'),
                       ('nseq', 'U5'),
                       ('nad1', 'U5'),
                       ('nad2', 'U5'),
                       ('isot', np.int32),
                       ('d1', np.float64),
                       ('d2', np.float64),
                       ('areax', np.float64),
                       ('betax', np.float64),
                       ('sigx', np.float64), ])
        data_conne = np.empty(len(self.connections), dtype=dt)

        for i_conn, conn in enumerate(self.connections):
            data_conne['name1'][i_conn] = conn.name1
            data_conne['name2'][i_conn] = conn.name2
            data_conne['nseq'][i_conn] = conn.nseq
            data_conne['nad1'][i_conn] = conn.nad1
            data_conne['nad2'][i_conn] = conn.nad2
            data_conne['isot'][i_conn] = conn.isot
            data_conne['d1'][i_conn] = conn.d1
            data_conne['d2'][i_conn] = conn.d2
            data_conne['areax'][i_conn] = conn.areax
            data_conne['betax'][i_conn] = 0.0 if conn.betax == None else conn.betax
            data_conne['sigx'][i_conn] = 0.0 if conn.sigx  == None else conn.sigx

        return data_conne

    def update_from_numpy_array(self, data_conne, col_names=None, idx=None):

        if idx is None:
            # Update all connections:
            idx = np.arange(len(self.connections))
        if col_names is None:
            # Update all columns
            col_names = ['name1', 'name2', 'nseq', 'nad1', 'nad2', 'isot', 'd1', 'd2', 'areax', 'betax', 'sigx']

        if len(data_conne) != len(self.connections):
            # At least one connection has been added or removed, so regenerate new list of Eleme objects
            print('The length of the numpy array data_conne does not match the number of connections in the mesh'
                  'object.')
            exit()

        else:
            # No connections were added or removed.  Only the attributes listed in col_names in entires from idx
            # list have been changed
            for i_con in idx:
                for col_name in col_names:
                    # Update only columns provided in col_names
                    setattr(self.connections[i_con], col_name, data_conne[col_name][i_con])

        return None

    def to_file(self, f):
        # Print CONNE list to file 'f' (either filename or handle)
        hdr = 'CONNE----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        data_conne = self.as_numpy_array()
        fmt = ['%5s', '%5s', '%5s', '%5s', '%5s', '%5u', '%10.4E', '%10.4E', '%10.4E', '%10.4f', '%10.4E']
        np.savetxt(f, data_conne, header=hdr, delimiter='', fmt=fmt, comments='')

        return None

def gen_connections(fname):
    """ read the fname and parse the connection data

    For a description of what the parameters mean, see the TOUGH2 manual,
    page 173.

    """
    with open(fname, 'r') as f:
        idx =  None
        for line in f:
            if "CONNE" in line:
                idx = 0
                continue
            if idx is not None:
                if line.strip() == "":
                    break
                name = line[0:5]
                name2 = line[5:10]
                nseq = line[10:15]
                nad1 = line[15:20]
                nad2 = line[20:25]
                isot = int(line[25:30])
                d1 = float_or_none(line[30:40])
                d2 = float_or_none(line[40:50])
                areax = float_or_none(line[50:60])
                betax = float_or_none(line[60:70])
                sigx = float_or_none(line[70:80])
                yield Conne(name, name2, nseq, nad1, nad2, isot, d1, d2, areax, betax, sigx)
                idx+=1

if __name__ == '__main__':

    from os.path import dirname as up

    # base_dir = os.path.join(os.pardir, 'test_data')
    base_dir = os.path.join(up(up(up(os.getcwd()))), 'output')
    fname = os.path.join(base_dir, 'MESHF')
    pck_file = fname + '.pck'
    mesh = Mesh.from_pickle(pck_file)

    idx_aek70 = mesh.nodes.name2idx['AEK70']
    bottom_gener_el = mesh.nodes[idx_aek70]
    connected_el_inds = bottom_gener_el.connected_elements
    gener_con_inds = bottom_gener_el.connections
    isn1_inds = bottom_gener_el.is_n1
    beta_list = []
    isot_list = []
    con_els_names_list = []

    for ci, cei in zip(gener_con_inds, connected_el_inds):
        con_els_names_list.append(mesh.nodes[cei].name)
        conn = mesh.connections[ci]
        beta_list.append(conn.betax)
        isot_list.append(conn.isot)

    print(con_els_names_list)
    print(isn1_inds)
    print(beta_list)
    print(isot_list)

    exit()

#    pdir = os.path.join(".", "test_data_stomp")
#    mesh_filename = 'MESH_in'
#    pck_file = os.path.join(pdir, 'temp_mesh.pck')

    #mesh_file = os.path.join(pdir, "SMA3Dn_R09_1_gas.mesh", "SMA3Dn_R09_1_gas.mesh")

    # test_incon()

#    exit()

    try:
        mesh = Mesh.from_pickle(pck_file)
        # raise Exception
        print("Loading the mesh from a pickle ")
    except Exception as e:
        mesh_file = os.path.join(pdir, mesh_filename)
        mesh = Mesh(mesh_file)
        mesh.to_pickle(pck_file)
        print("Pickled the mesh ")
