"""
Parse a TOUGH2 MESH file

For python 3.4
"""
import ast
import os
# import pickle
import numpy as np
import io
# import pandas as pd


def type_or_none(s, data_type='str'):
    """ converts a string to a given data type (data_type); if it is empty, it will return None"""
    try:
        if data_type == 'str':
            if not s.strip():
                s = None
            else:
                s = s.strip()
            return s
        elif data_type == 'int':
            return int(s)
        elif data_type == 'float':
            return float(s)
    except ValueError:
        return None


class TOUGHInput:

    def __init__(self, title, blocks=None):
        self.title = title
        if blocks is None:
            self.blocks = []
        else:
            self.blocks = blocks

    def append(self, block):
        self.blocks.append(block)

        return None

    def to_file(self, fn):
        f = open(fn, 'w')

        if type(self.title) == str:
            f.write(self.title)
        elif type(self.title) == list:
            for title_line in self.title:
                f.write(title_line + '\n')

        data_str = ''
        for block in self.blocks:
            data_str += '\n'
            data_str += block.to_file()

        f.write(data_str)
        f.close()

        return None

    @property
    def keyword_list(self):
        keyword_list = []
        if self.blocks is not None:
            for block in self.blocks:
                keyword_list.append(block.keyword)
        return keyword_list

    @classmethod
    def from_file(cls, fn):

        title = ''
        eligible_keywords = TOUGHBlock.get_eligible_keywords()
        f = open(fn, 'r')
        f_data = f.readlines()
        blocks = []

        while len(f_data) > 0:
            line = f_data[0]
            if line[:5] in eligible_keywords:
                # Start filling in block info.
                keyword = line[:5]
                block_cls_str = keyword[0] + keyword[1:].lower()
                lines, i_lines = globals()[block_cls_str].find_lines_from_list(f_data, return_indices=True)
                block = globals()[block_cls_str].block_from_lines(lines)
                blocks.append(block)
                f_data = f_data[i_lines[-1]+1:]
            else:
                # Title line(s):
                if type(title) is str:
                    if title == '':
                        title = line.strip('\n')
                    else:
                        title = [title, line.strip('\n')]
                elif type(title) is list:
                    title.append(line.strip('\n'))
                f_data = f_data[1:]

        return cls(title, blocks=blocks)


class TOUGHBlock:

    def __init__(self, record_collections=None, trc=None, end_with_blank_line=False):
        self.keyword = type(self).__name__[:5].upper()
        self.eligible_keywords = self.get_eligible_keywords()
        if record_collections is None:
            self.record_collections = []
        else:
            self.record_collections = record_collections
        self.end_with_blank_line = end_with_blank_line
        if trc is None:
            self.trc = TOUGHRecordCollection
        else:
            self.trc = trc

    @classmethod
    def get_eligible_keywords(cls):
        return ['MESHM', 'ROCKS', 'RCPAP', 'MULTI', 'START', 'PARAM', 'INDOM', 'INCON', 'SOLVR', 'FOFT ',
                'COFT ', 'GOFT ', 'NOVER', 'DIFFU', 'SELEC', 'RPCAP', 'TIMES', 'ELEME', 'CONNE', 'GENER',
                'MOMOP', 'REACT', 'OUTPU', 'ENDFI', 'ENDCY']

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False, return_line_indices=False):
        # This routine reads block information from a file.  First, it finds the lines of the file containing pertinent
        # block data.  Then, it parses those lines for each record collection until all lines have been read.

        if return_line_indices:
            lines, ind_lines = cls.find_lines(fn, end_with_blank_line=end_with_blank_line, return_indices=True)
        else:
            ind_lines = None
            lines = cls.find_lines(fn, end_with_blank_line=end_with_blank_line)
        # # Check for blocks (like ROCKS) that end with blank lines but could have an empty line before end of block
        # if (not end_with_blank_line) and (not lines or lines[-1].strip() == ''):
        #     # end_with_blank_line indicates False, but last line is blank, so remove last line
        #     lines = lines[:-1]
        if return_line_indices:
            # User requested the line indices where block appears in file
            return cls.block_from_lines(lines, trc=trc), ind_lines
        else:
            return cls.block_from_lines(lines, trc=trc)

    @classmethod
    def find_lines(cls, fn, end_with_blank_line=False, return_indices=False):
        # This method reads in all records from a TOUGH Block and pulls in all text lines need to generate
        # a TOUGHBlock object.
        f = open(fn, 'r')
        lines_list = f.readlines()
        return cls.find_lines_from_list(lines_list, end_with_blank_line, return_indices=return_indices)

    @classmethod
    def find_lines_from_list(cls, lines_list, end_with_blank_line=False, return_indices=False):

        keyword = cls.__name__[:5].upper()
        eligible_keywords = cls.get_eligible_keywords()
        lines = []
        i_lines = []
        read_lines = False

        for i_line, line in enumerate(lines_list):
            if line[:len(keyword)] == keyword:
                i_lines.append(i_line)
                read_lines = True
                continue
            if read_lines:
                if line[:5] in eligible_keywords:
                    break
                elif end_with_blank_line and not line.strip():
                    break
                elif cls.__base__.__name__ == 'TOUGHZeroLineBlock':
                    break
                else:
                    i_lines.append(i_line)
                    lines.append(line)

        # Check for blocks (like ROCKS) that end with blank lines but could have an empty line before end of block
        if (not end_with_blank_line) and (not lines or lines[-1].strip() == ''):
            # end_with_blank_line indicates False, but last line is blank, so remove last line
            lines = lines[:-1]

        if return_indices:
            return lines, i_lines
        else:
            return lines

    @classmethod
    def block_from_lines(cls, lines, trc=None):

        record_collections = []
        if trc is None:
            trc = TOUGHRecordCollection

        while len(lines) > 0:
            # Parse lines for data for each record collection. Return a record collection object and the remaining
            # lines to be read.
            rc, lines = trc.from_file(lines)
            record_collections.append(rc)

        return cls(record_collections)

    def __getitem__(self, item):
        if self.record_collections is None:
            return None
        else:
            if len(self.record_collections) == 1:
                return self.record_collections[0].records[item]
            elif len(self.record_collections) > 1:
                return self.record_collections[item]

    def __setitem__(self, item, data):
        if self.record_collections is not None:
            self.record_collections[item] = data

        return None

    def append(self, record):
        self.record_collections.append(record)

        return None

    def to_file(self, fname=None, update_records=True, prepend_line=None):
        block_str = ''
        if prepend_line is not None:
            block_str += prepend_line + '\n'
        block_str += self.keyword + '----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        if type(self.record_collections) == list:
            rcs = self.record_collections
        else:
            rcs = [self.record_collections]
        for rc in rcs:
            block_str += rc.to_file(update_records=update_records)

        if self.end_with_blank_line:
            block_str += '\n'

        if fname is not None:
            f = open(fname, 'w')
            f.write(block_str)
            f.close()

        return block_str

    def activate(self):
        self.keyword = self.keyword.upper()

    def deactivate(self):
        self.keyword = self.keyword[0].lower() + self.keyword[1:]

    def elevate_attributes(self):
        return None


class TOUGHSimpleBlock(TOUGHBlock):

    # This is a subset of TOUGH blocks that contain only a single TOUGHRecordCollection

    def __init__(self, record_collections=None, trc=None, end_with_blank_line=False):
        super().__init__(record_collections=record_collections, trc=trc, end_with_blank_line=end_with_blank_line)
        self.names = []

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        if trc is None:
            trc = TOUGHRecordCollection
        if return_line_indices:
            block, ind_lines = super().from_file(fn, trc=trc, return_line_indices=True)
        else:
            ind_lines = None
            block = super().from_file(fn, trc=trc, return_line_indices=False)
        block.names = [] if names is None else names
        # block.elevate_attributes()

        if return_line_indices:
            return block, ind_lines
        else:
            return block

    @classmethod
    def block_from_lines(cls, lines, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        block = super().block_from_lines(lines, trc=trc)
        block.names = names
        block.elevate_attributes()

        return block

    def to_file(self, fname=None, update_records=True, prepend_line=None):
        for name in self.names:
            setattr(self.record_collections[0], name, getattr(self, name))
        return super().to_file(fname=fname, update_records=update_records, prepend_line=prepend_line)

    def fill_attributes(self, **kwargs):
        # Sets attributes using keys and values of keyword arguments dictionary
        for i_arg in kwargs:
            setattr(self, i_arg, kwargs[i_arg])

        return None

    def elevate_attributes(self):
        # Elevates attributes from first (and only) record collection to the block level and deletes the attributes
        # from the record collection
        for name in self.names:
            setattr(self, name, getattr(self.record_collections[0], name))
            delattr(self.record_collections[0], name)

        return None

    def trc_from_args(self, *args):
        self.record_collections = [self.trc(*args)]
        return None


class TOUGHOneLineBlock(TOUGHBlock):

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False, return_line_indices=False):
        if return_line_indices:
            block, ind_lines = super().from_file(fn, trc=trc, end_with_blank_line=end_with_blank_line,
                                                 return_line_indices=True)
        else:
            ind_lines = None
            block = super().from_file(fn, trc=trc, end_with_blank_line=end_with_blank_line)
        for name in block.record_collections[0].names:
            # Pull attributes from record collection up to block level
            setattr(block, name, getattr(block.record_collections[0], name))
        if return_line_indices:
            return block, ind_lines
        else:
            return block

    def to_file(self, fname=None, update_records=True, prepend_line=None):
        if update_records:
            if type(self.record_collections) == list:
                names = self.record_collections[0].names
            else:
                names = self.record_collections.names
            # Push block level attributes down to record collection:
            for name in names:
                if type(self.record_collections) == list:
                    setattr(self.record_collections[0], name, getattr(self, name))
                else:
                    setattr(self.record_collections, name, getattr(self, name))

        return super().to_file(fname=fname, update_records=update_records, prepend_line=prepend_line)


class TOUGHZeroLineBlock(TOUGHBlock):

    def to_file(self, fname=None, update_records=True, prepend_line=None):

        block_str = self.keyword + '----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'

        if fname is not None:
            f = open(fname, 'w')
            f.write(block_str)
            f.close()

        return block_str


class TOUGHRecordCollection:

    def __init__(self, records=None):
        self.records = []
        self.names = []
        if records is not None:
            for record in records:
                self.append(record)

    @classmethod
    def empty(cls):
        return TOUGHRecordCollection()

    @classmethod
    def from_file(cls, data_record):

        rc = cls.empty()
        reject_list = rc.records[0].update_from_file(data_record)
        for entry in rc.records[0].entries:
            if entry.name is not None:
                setattr(rc, entry.name, entry.value)
        return rc, reject_list

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        if self.records is None:
            return None
        else:
            if len(self.records) == 1:
                return self.records[0].entries[item]
            elif len(self.records) > 1:
                return self.records[item]

    def __setitem__(self, item, data):

        if self.records is not None:
            if len(self.records) == 1:
                # Set entry ('data' is a TOUGHEntry object) in only record in record collection
                if self.records[0].entries[item] is not None:
                    # Delete previously listed entry
                    delattr(self, self.records[0].entries[item].name)
                # Set entry in (only) record listing
                self.records[0].entries[item] = data
                # Generate attribute in only record listed
                if data[0] is not None:
                    setattr(self, data[0], data[1])

            elif len(self.records) > 1:
                # Set record ('data' is a TOUGHRecord object)
                if self.records[item] is not None:
                    for entry in self.records[item].entries:
                        # Delete attributes in previously listed record
                        delattr(self, entry.name)
                self.records[item] = data
                for entry in data.entries:
                    # Add new attribute in record
                    if entry.name is not None:
                        setattr(self, entry.name, entry.value)

        return None

    def append(self, record):
        # Add a record, which is a list of entries
        self.records.append(record)
        for entry in record.entries:
            # Populate attributes of record collection based on names and values of entries
            if entry.name is not None:
                setattr(self, entry.name, entry.value)

        return None

    def fill_attributes(self):
        for rec in self.records:
            for entry in rec.entries:
                setattr(self, entry.name, entry.value)
        return None

    def to_file(self, update_records=True):

        if update_records:
            self.update_records()
        record_str = ''
        for record in self.records:
            record_str += '\n'
            record_str += record.to_file()

        return record_str

    def append_from_file(self, data_record, data_fmt):
        self.append(TOUGHRecord.from_file(data_record, data_fmt))
        return None

    def update_records(self, *args, **kwargs):

        self.records = []
        if not args and not kwargs:
            args = ()
            for name in self.names[0]:
                if name is not None and hasattr(self,name):
                    args += (getattr(self, name), )

            kwargs = {}
            for names in self.names[1:]:
                for name in names:
                    if name is not None and hasattr(self, name):
                        kwargs.update({name: getattr(self, name)})

            args, kwargs = self.name_to_args(args, kwargs)

        return args, kwargs

    def name_to_args(self, args, kwargs):
        return args, kwargs

    def add_tables(self, table_names, entries_per_record, data_fmt):
        # Add table data:
        for tab_name in table_names:
            remaining_data = getattr(self, tab_name)
            i_table = 1
            while len(remaining_data) > 0:
                record = TOUGHRecord()
                record_data, remaining_data, _ = np.split(remaining_data, [entries_per_record, len(remaining_data)])
                for rd in record_data:
                    record.append((None, rd, data_fmt))
                    i_table += 1
                self.append(record)

        return None

    def read_table(self, lines, num_entries, entries_per_record, data_fmt, data_type='float'):
        # This routine reads in a table of num_entry values from lines of a text file whose lines contain no more than
        # entries_per_record values per line.  It appends records the object, and outputs both an array of table data
        # and any remaining lines from the block of text provided to be read.
        if num_entries is None:
            num_lines = len(lines)
            len_entry = len(data_fmt.format(0))
            num_entries = int(((num_lines-1)*entries_per_record +
                              1 + int(np.ceil(len(lines[-1][len_entry:].strip())/len_entry))))
        else:
            num_lines = int(np.ceil(num_entries / entries_per_record))
        if data_type == 'int':
            dt = int
        else:
            dt = float
        tab_data = np.empty(num_entries, dtype=dt)
        remaining_terms = num_entries  # Track how many remaining entries must be read
        for i_line, line in enumerate(lines[:num_lines]):
            # Number of entries that will be read from this line
            terms_to_read = min(remaining_terms, entries_per_record)
            data_fmts = [(None, data_fmt, data_type)] * terms_to_read
            self.append_from_file(line, data_fmts)
            tab_data[i_line * entries_per_record:i_line * entries_per_record + terms_to_read] = self[-1].get_data()
            remaining_terms -= terms_to_read  # Update the number of remaining entries to read

        if any(np.isnan(tab_data)):
            tab_data[np.isnan(tab_data)] = None

        return tab_data.tolist(), lines[num_lines:]


class TOUGHRecord:

    def __init__(self, entries=None):
        if entries is None:
            self.entries = []
        else:
            self.entries = entries
            for entry in self.entries:
                if entry.name is not None:
                    setattr(self, entry.name, entry.value)

    @classmethod
    def from_file(cls, data_record, data_fmt):
        # Build a new record based on a record (line) from a text file (data_record) and a list of tuples (data_fmt)
        # containing the following information:
        # 1. data_fmt[i][0] - The name of the entry/variable
        # 2. data_fmt[i][1] - The data format string (if printed to an output file, e.g., '{:>14.7E}'
        # 3. data_fmt[i][2] - The data type of the entry ('str', 'float', or 'int')
        rec_offset = 0
        rec = TOUGHRecord()
        for item in np.arange(len(data_fmt)):
            data_width = len(data_fmt[item][1].format(0))
            rec.entries.append(TOUGHEntry.from_file(data_record[rec_offset:rec_offset+data_width], data_fmt[item]))
            rec_offset += data_width

        return rec

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        return self.entries[item]

    def __setitem__(self, item, data):
        self.entries[item] = TOUGHEntry(*data)
        if data[0] is not None:
            setattr(self, data[0], data[1])

        return None

    def append(self, entries_data):
        # Add data from a single entry (as a tuple) or from multiple entries (as a list of tuples)

        if type(entries_data) == list:
            # Inputs are data from multiple entries
            for entry_data in entries_data:
                # Generate TOUGHEntry object with entry_data and add to entry list
                self.entries.append(TOUGHEntry(*entry_data))
                # Add attribute to record based on name (entry_data[0]) and value (entry_data[1])
                if entry_data[0] is not None:
                    setattr(self, entry_data[0], entry_data[1])
        elif type(entries_data) == tuple:
            # Generate object with entry_data and add to entry list
            self.entries.append(TOUGHEntry(*entries_data))
            # Add attribute to record based on name (entry_data[0]) and value (entry_data[1])
            if entries_data[0] is not None:
                setattr(self, entries_data[0], entries_data[1])

        return None

    def get_data(self):
        data = []
        for entry in self.entries:
            data.append(entry.value)

        return data

    def to_file(self):
        record_str = ''
        for entry in self.entries:
            record_str += entry.to_file()

        return record_str

    def update_from_file(self, data_record):
        rec_offset = 0
        reject_list = []
        if type(data_record) == list:
            data_record = data_record[0]
        for item in np.arange(len(self)):
            entry_width = len(self[item])
            self[item].update_from_file(data_record[rec_offset:rec_offset + entry_width])
            if self[item].name is not None:
                setattr(self, self[item].name, self[item].value)
            else:
                reject_list.append(self[item].value)
            rec_offset += entry_width

        return reject_list


class TOUGHEntry:

    def __init__(self, name, value, fmt_str):
        self.name = name
        self.value = value
        self.fmt_str = fmt_str

    @classmethod
    def from_file(cls, data_record, data_fmt):
        name = data_fmt[0]
        val = type_or_none(data_record, data_type=data_fmt[2])
        fmt_str = data_fmt[1]
        return TOUGHEntry(name, val, fmt_str)

    def __len__(self):
        return len(self.fmt_str.format(0))

    def to_file(self):
        if self.value is None:
            fmt_str = '{:>' + str(len(self)) + '}'
            return fmt_str.format('')
        else:
            fmt_str = self.fmt_str
            # if '.' in self.fmt_str and self.value > 0.0:
            #     # Find location of decimal in format string:
            #     i_dot = self.fmt_str.find('.')
            #     str_list = list(self.fmt_str)
            #     # Add one to precision if value is nonnegative
            #     str_list = str_list[:i_dot+1] + [str(int(str_list[i_dot+1])+1)] + str_list[i_dot+2:]
            #     fmt_str = ''.join(str_list)
            if 'E' in self.fmt_str:
                fmt_str = fmt_str.replace('E', '')
                out_str = fmt_str.format(self.value).upper()
                if ('E' in out_str) and not ('.' in out_str):
                    out_str = out_str[2:]
                    e_loc = out_str.find('E')
                    out_str = out_str[:e_loc] + '.0' + out_str[e_loc:]
                return out_str
            elif 'f' in self.fmt_str:
                fmt_str = fmt_str.replace('f','')
                return fmt_str.format(self.value).upper()
            else:
                return fmt_str.format(self.value)

    def update_from_file(self, data_record):
        self.value = type_or_none(data_record, data_type=type(self.value).__name__)
        return None


class Rocks(TOUGHBlock):

    def __init__(self, rocks=None):
        super().__init__(record_collections=rocks, trc=Rock, end_with_blank_line=True)

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False, return_line_indices=False):
        return super().from_file(fn, trc=Rock, end_with_blank_line=end_with_blank_line,
                                 return_line_indices=return_line_indices)

    @classmethod
    def block_from_lines(cls, lines, trc=None):
        return super().block_from_lines(lines, trc=Rock)


class Rock(TOUGHRecordCollection):
    """ represents a term in the ROCKS block """

    def __init__(self, mat, drok, por, per, cwet, spht,
                 com=None, expan=None, cdry=None, tortx=None, gk=None, xkd3=None, xkd4=None,
                 irp=None, rp=None, icp=None, cp=None):

        super().__init__()
        self.names = [['mat', 'nad', 'drok', 'por', None, None, None, 'cwet', 'spht'],
                      ['com', 'expan', 'cdry', 'tortx', 'gk', 'xkd3', 'xkd4'],
                      ['irp', 'rp', 'icp', 'cp']]
        args = (mat, drok, por, per, cwet, spht)
        kwargs = {'com':com, 'expan':expan, 'cdry':cdry, 'tortx': tortx, 'gk':gk, 'xkd3':xkd3, 'xkd4':xkd4,
                  'irp':irp, 'rp':rp, 'icp':icp, 'cp':cp}
        self.update_records(*args, **kwargs)

    @classmethod
    def empty(cls):
        super().empty()
        return Rock('NTRCK', 0.0, 0.0, [0.0, 0.0, 0.0], 0.0, 0.0)

    @classmethod
    def from_file(cls, data_record):
        rock, per = super().from_file(data_record[0])
        setattr(rock, 'per', per)
        num_records = 1
        nad = getattr(rock, 'nad')
        if nad is not None and nad >= 1:
            num_records += 1
            names = getattr(rock, 'names')[1]
            data_fmt = []
            for name in names:
                data_fmt.append((name, '{:>10.4E}', 'float'))
            rock.append(TOUGHRecord.from_file(data_record[1], data_fmt))
            if nad >= 2:
                num_records += 2
                rpcp, _ = RpCp.from_file(data_record[2:4])
                setattr(rock, 'rp', getattr(rpcp,'rp'))
                setattr(rock, 'cp', getattr(rpcp,'cp'))
                rock.append(rpcp[0])
                rock.append(rpcp[1])

        return rock, data_record[num_records:]

    def update_records(self, *args, **kwargs):
        """Updates records based on either inputs provided by user or (if no inputs are provided) from class
        attributes."""

        args, kwargs = super().update_records(*args, **kwargs)

        if all(list(kwargs.values())[7:]):
            # RP and CAP functions are provided for this rock type
            nad = 2
        elif any(list(kwargs.values())[:7]):
            # Only one additional record is required
            nad = 1
        else:
            nad = 0
        setattr(self, 'nad', nad)
        record = TOUGHRecord()
        record.append([('mat', args[0], '{:<5}'),
                       ('nad', nad, '{:>5}'),
                       ('drok', args[1], '{:>10.4f}'),
                       ('por', args[2], '{:>10.4f}'),
                       (None, args[3][0], '{:>10.4E}'),
                       (None, args[3][1], '{:>10.4E}'),
                       (None, args[3][2], '{:>10.4E}'),
                       ('cwet', args[4], '{:>10.4f}'),
                       ('spht', args[5], '{:>10.4E}')])
        self.append(record)
        setattr(self, 'per', args[3])

        if nad >= 1:
            self.add_special_records(**kwargs)
        else:
            # Fill in None values for all remaining attributes
            names = ['com', 'expan', 'cdry', 'tortx', 'gk', 'xkd3', 'xkd4', 'irp', 'rp', 'icp', 'cp']
            for name, value in zip(names, list(kwargs.values())):
                setattr(self, name, value)

        return None

    def add_special_records(self, com, expan, cdry, tortx, gk, xkd3, xkd4, irp=None, rp=None, icp=None, cp=None):

        record = TOUGHRecord()
        data_fmt = '{:>10.4E}'
        record.append([('com', com, data_fmt),
                       ('expan', expan, data_fmt),
                       ('cdry', cdry, data_fmt),
                       ('tortx', tortx, data_fmt),
                       ('gk', gk, data_fmt),
                       ('xkd3', xkd3, data_fmt),
                       ('xkd4', xkd4, data_fmt)])
        self.append(record)

        nad = getattr(self, 'nad')
        if nad >= 2:
            for nam, ircp, rcp in zip(['rp', 'cp'], [irp, icp], [rp, cp]):
                record = TOUGHRecord()
                record.append([('i' + nam, ircp, '{:>5}'),
                               (None, None, '{:>5}')])
                for rc in rcp:
                    record.append((None, rc, '{:>10.4E}'))
                self.append(record)
                setattr(self, nam, rcp)
        else:
            # Fill in None values for irp, rp, icp, and cp
            for name in self.names[1]:
                setattr(self, name, None)

        return None

    def name_to_args(self, args, kwargs):

        # Remove nad entry (which is in self.names) from args (which does not contain nad)
        args = list(args)
        del args[1]
        # Insert per entry (which is not in self.names) into args (which contains per)
        args.insert(3, getattr(self, 'per'))
        args = tuple(args)

        return args, kwargs


class Rpcap(TOUGHBlock):

    def __init__(self, rpcp=None):

        super().__init__(record_collections=rpcp)
        self.names = [['irp'], ['icp']]

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False, return_line_indices=False):
        return super().from_file(fn, trc=RpCp, end_with_blank_line=end_with_blank_line,
                                 return_line_indices=return_line_indices)

    @classmethod
    def block_from_lines(cls, lines, trc=None):
        return super().block_from_lines(lines, trc=RpCp)

def get_rpcp_list(data_record_snippet):

    # Create a list of RPCAP variables given snippet from input file
    data_record_snippet = data_record_snippet.rstrip()
    rpcp_list = []

    while len(data_record_snippet) > 0:
        data_entry = data_record_snippet[:10]
        if data_entry.strip() == '':
            rpcp_list.append(None)
        else:
            rpcp_list.append(float(data_entry))
        data_record_snippet = data_record_snippet[10:]

    return rpcp_list


class RpCp(TOUGHRecordCollection):

    def __init__(self, irp, rp, icp, cp):

        super().__init__(records=None)
        self.update_records(irp, rp, icp, cp)

    def update_records(self, *args, **kwargs):

        if not args:
            self.records = []
            self.append(RelPerm(getattr(self, 'irp'), getattr(self, 'rp')))
            self.append(CapPress(getattr(self, 'icp'), getattr(self, 'cp')))
        else:
            self.append(RelPerm(args[0], args[1]))
            self.append(CapPress(args[2], args[3]))
            setattr(self, 'rp', args[1])
            setattr(self, 'cp', args[3])

        return None

    @classmethod
    def from_file(cls, data_record):

        irp = int(data_record[0][:5])
        rp = get_rpcp_list(data_record[0][10:])
        # rp = [float(rp_i) for rp_i in data_record[0][10:].split()]

        icp = int(data_record[1][:5])
        cp = get_rpcp_list(data_record[1][10:])
        # cp = [float(cp_i) for cp_i in data_record[1][10:].split()]

        return RpCp(irp, rp, icp, cp), []


class RelPerm(TOUGHRecord):

    def __init__(self, irp, rp):

        super().__init__()
        self.append([('irp', irp, '{:>5}'), (None, None, '{:>5}')])
        for r in rp:
            self.append((None, r, '{:>10.4E}'))


class CapPress(TOUGHRecord):

    def __init__(self, icp, cp):

        super().__init__()
        self.append([('icp', icp, '{:>5}'), (None, None, '{:>5}')])
        for c in cp:
            self.append((None, c, '{:>10.4E}'))





class Multi(TOUGHSimpleBlock):

    def __init__(self, multi=None, nk=None, neq=None, nph=2, nb=6, nkin=None):
        super().__init__(record_collections=multi, trc=Mult)
        if multi is None:
            self.append(Mult(nk, neq=neq, nph=nph, nb=nb, nkin=nkin))
            self.nk = nk
            self.neq = neq
            self.nph = nph
            self.nb = nb
            self.nkin = nkin

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        names = ['nk', 'neq', 'nph', 'nb', 'nkin']
        block = super().from_file(fn, names=names, trc=Mult, return_line_indices=return_line_indices)
        return block

    @classmethod
    def block_from_lines(cls, lines, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        names = ['nk', 'neq', 'nph', 'nb', 'nkin']
        return super().block_from_lines(lines, names=names, trc=Mult, return_line_indices=return_line_indices)


class Mult(TOUGHRecordCollection):

    def __init__(self, nk, neq=None, nph=2, nb=6, nkin=None):
        super().__init__()
        self.names = ['nk', 'neq', 'nph', 'nb', 'nkin']
        self.update_records(nk, neq, nph, nb, nkin)

    def update_records(self, *args, **kwargs):

        data_fmt = '{:>5}'
        entries_data = []
        if not args:
            self.records = []
            for name in self.names:
                entries_data.append((name, getattr(self, name), data_fmt))
        else:
            if args[1] is None:
                args = list(args)
                args[1] = args[0]
                args = tuple(args)
            for arg, name in zip(args, self.names):
                entries_data.append((name, arg, data_fmt))
        rec = TOUGHRecord()
        rec.append(entries_data)
        self.append(rec)

        return None

    @classmethod
    def empty(cls):
        return Mult(0)


class Momop(TOUGHBlock):

    def __init__(self, mop2=None, mop2_list=None):
        super().__init__(record_collections=mop2, trc=Mop2)
        self.names = ['mop2']
        if mop2 is None:
            self.append(Mop2(mop2_list))
            setattr(self, 'mop2', mop2_list + (27-len(mop2_list))*[None])

    @classmethod
    def from_file(cls, fn, trc = None, end_with_blank_line=False, return_line_indices=False):
        block = super().from_file(fn, Mop2, end_with_blank_line, return_line_indices)
        return block

    @classmethod
    def block_from_lines(cls, lines, trc=None):
        return super().block_from_lines(lines, trc=Mop2)


class Mop2(TOUGHRecordCollection):

    def __init__(self, mop2_list):
        super().__init__()
        self.names = ['mop2']
        self.update_records(mop2_list)

    def update_records(self, *args, **kwargs):

        if not args:
            self.records = []
        else:
            setattr(self, self.names[0], args[0])

        self.add_tables(['mop2'], entries_per_record=80, data_fmt='{:>1}')
        mop2_list = getattr(self, 'mop2')
        if len(mop2_list) < 27:
            setattr(self, 'mop2', mop2_list + [None]*(27-len(mop2_list)))

        return None

    @classmethod
    def empty(cls):
        return Mop2([0]*27)

    @classmethod
    def from_file(cls, data_record):
        rc = cls.empty()
        mop2 = list(data_record[0][:-1])
        for i_m, m in enumerate(mop2):
            if m == ' ':
                mop2[i_m] = None

        setattr(rc, 'mop2', mop2 + [None]*(27 - len(mop2)))

        return rc, []


class React(TOUGHOneLineBlock):

    def __init__(self, mopr=None, mopr_list=None):
        super().__init__(record_collections=mopr, trc=Mopr)
        self.names = ['mopr']
        if mopr is None:
            self.append(Mopr(mopr_list))
            setattr(self, 'mopr', mopr_list + (20-len(mopr_list))*[None])

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False):
        block = super().from_file(fn, trc=Mopr)
        return block


class Mopr(TOUGHRecordCollection):

    def __init__(self, mopr_list):
        super().__init__()
        self.names = ['mopr']
        self.update_records(mopr_list)

    def update_records(self, *args, **kwargs):

        if not args:
            self.records = []
        else:
            setattr(self, self.names[0], args[0])

        self.add_tables(['mopr'], entries_per_record=80, data_fmt='{:>1}')
        mopr_list = getattr(self, 'mopr')
        if len(mopr_list) < 20:
            setattr(self, 'mopr', mopr_list + [None]*(20-len(mopr_list)))

        return None

    @classmethod
    def empty(cls):
        return Mopr([0]*20)

    @classmethod
    def from_file(cls, data_record):
        rc = cls.empty()
        mopr = list(data_record[0][:-1])
        for i_m, m in enumerate(mopr):
            if m == ' ':
                mopr[i_m] = None

        setattr(rc, 'mopr', mopr + [None]*(20 - len(mopr)))

        return rc, []


class Times(TOUGHSimpleBlock):

    def __init__(self, times_collection=None, ite=None, delaf=None, tinter=None, tis=None):
        super().__init__(record_collections=times_collection, trc=TimesCollection)
        self.names = ['iti', 'ite', 'delaf', 'tinter', 'tis']
        if times_collection is None:
            iti = 0 if tis is None else len(tis)
            self.fill_attributes(iti=iti, ite=ite, delaf=delaf, tinter=tinter, tis=tis)
            self.trc_from_args(ite, delaf, tinter, tis)

    def __getitem__(self, item):
        return getattr(self, 'tis')[item]

    def __setitem__(self, key, value):
        tis = getattr(self, 'tis')
        tis[key] = value
        setattr(self,'tis', tis)
        return None

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False):
        return super().from_file(fn, trc=TimesCollection, names=['iti', 'ite', 'delaf', 'tinter', 'tis'])


class TimesCollection(TOUGHRecordCollection):

    def __init__(self, ite=None, delaf=None, tinter=None, tis=None):
        super().__init__()
        self.names = [['ite', 'delaf', 'tinter'],
                      ['tis']]
        self.tis = tis
        self.update_records(ite, delaf, tinter, tis)

    def update_records(self, *args, **kwargs):

        rec = TOUGHRecord()
        if not args and not kwargs:
            self.records = []
            args = []
            for names in self.names:
                for name in names:
                    args.append(getattr(self, name))
            args = tuple(args)

        iti = 0 if args[3] is None else len(args[3])
        rec.append([('iti', iti, '{:>5}'),
                    ('ite', args[0], '{:>5}'),
                    ('delaf', args[1], '{:>10.4E}'),
                    ('tinter', args[2], '{:>10.4E}')])

        self.append(rec)
        self.tis = args[3]

        if iti > 0:
            table_names = ['tis']
            self.add_tables(table_names, 8, '{:>10.4E}')

        return None

    @classmethod
    def empty(cls):
        return TimesCollection(0, 0.0, 0.0)

    @classmethod
    def from_file(cls, data_record):
        times_collection, reject_list = super().from_file(data_record[0])
        iti = getattr(times_collection, 'iti')
        if iti is not None and iti > 0:
            # Read in table of times values
            times_collection.tis, lines = times_collection.read_table(data_record[1:], iti, 8, '{:>10.4E}')
            return times_collection, []
        else:
            return times_collection, []


class Start(TOUGHZeroLineBlock):
    pass


class Endfi(TOUGHZeroLineBlock):
    pass


class Endcy(TOUGHZeroLineBlock):
    pass


class Outpu(TOUGHBlock):

    def __init__(self, outputrequests=None, coutfm=None):

        super().__init__(record_collections=outputrequests, trc=OutputRequests)
        self.coutfm = coutfm

    @property
    def moutvar(self):
        return len(self.record_collections)

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=False, return_line_indices=False):
        return super().from_file(fn, trc=OutputRequests, end_with_blank_line=end_with_blank_line,
                                 return_line_indices=return_line_indices)

    @classmethod
    def block_from_lines(cls, lines, trc=None):
        return super().block_from_lines(lines, trc=OutputRequests)


class OutputRequests(TOUGHRecordCollection):

    def __init__(self, output_requests, ):

        super().__init__(records=output_requests)


class OutputRequest(TOUGHRecord):

    def __init__(self, coutln, id1=None, id2=None):

        super().__init__()
        self.append([('coutln', coutln, '{:>20}'), ('id1', id1, '{:>5}'), ('id2', id2, '{:>5}')])


class Param(TOUGHSimpleBlock):

    def __init__(self, param_collection=None,
                 noite=None, kdata=None, mcyc=None, msec=None, mcypr=None, mop=None, texp=None, be=None,
                 tstart=None, timax=None, delten=None, deltmx=None, elst=None, gf=None, redlt=None, scale=None,
                 dlt=None,
                 re1=None, re2=None, wup=None, wnr=None, dfac=None,
                 dep=None):
        super().__init__(record_collections=param_collection, trc=ParamCollection)
        self.names = ['noite', 'kdata', 'mcyc', 'msec', 'mcypr', 'mop', 'texp', 'be',
                      'tstart', 'timax', 'delten', 'deltmx', 'elst', 'gf', 'redlt', 'scale',
                      'dlt',
                      're1', 're2', 'wup', 'wnr', 'dfac',
                      'dep']
        if param_collection is None:
            self.fill_attributes(noite=noite, kdata=kdata, mcyc=mcyc, msec=msec, mcypr=mcypr, mop=mop, texp=texp, be=be,
                                 tstart=tstart, timax=timax, delten=delten, deltmx=deltmx, elst=elst, gf=gf,
                                 redlt=redlt, scale=scale,
                                 dlt=dlt,
                                 re1=re1, re2=re2, wup=wup, wnr=wnr, dfac=dfac,
                                 dep=dep)
            self.trc_from_args(noite, kdata, mcyc, msec, mcypr, mop, texp, be,
                               tstart, timax, delten, deltmx, elst, gf,
                               redlt, scale,
                               dlt,
                               re1, re2, wup, wnr, dfac,
                               dep)
        #     self.append(ParamCollection(noite=noite, kdata=kdata, mcyc=mcyc, msec=msec, mcypr=mcypr,
        #                                 mop=mop, texp=texp, be=be, tstart=tstart, timax=timax,
        #                                 delten=delten, deltmx=deltmx, elst=elst, gf=gf, redlt=redlt,
        #                                 scale=scale, dlt=dlt, re1=re1, re2=re2, wup=wup, wnr=wnr,
        #                                 dfac=dfac, dep=dep))
        # else:
        #     self.elevate_attributes()

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        names = ['noite', 'kdata', 'mcyc', 'msec', 'mcypr', 'mop', 'texp', 'be',
                 'tstart', 'timax', 'delten', 'deltmx', 'elst', 'gf', 'redlt', 'scale',
                 'dlt',
                 're1', 're2', 'wup', 'wnr', 'dfac',
                 'dep']
        return super().from_file(fn, names=names, trc=ParamCollection, return_line_indices=return_line_indices)

    @classmethod
    def block_from_lines(cls, lines, names=None, trc=None, end_with_blank_line=False, return_line_indices=False):
        names = ['noite', 'kdata', 'mcyc', 'msec', 'mcypr', 'mop', 'texp', 'be',
                 'tstart', 'timax', 'delten', 'deltmx', 'elst', 'gf', 'redlt', 'scale',
                 'dlt',
                 're1', 're2', 'wup', 'wnr', 'dfac',
                 'dep']
        return super().block_from_lines(lines, names=names, trc=ParamCollection,
                                        return_line_indices=return_line_indices)

    def to_file(self, fname=None, update_records=True, prepend_line=None):
        pl = '----*----1--MOP:123456789*123456789*1234----*----5----*----6----*----7----*----8'
        return super().to_file(fname=fname, update_records=update_records, prepend_line=pl)


class ParamCollection(TOUGHRecordCollection):

    def __init__(self, noite=None, kdata=None, mcyc=None, msec=None, mcypr=None, mop=None, texp=None, be=None,
                 tstart=None, timax=None, delten=None, deltmx=None, elst=None, gf=None, redlt=None, scale=None,
                 dlt=None,
                 re1=None, re2=None, wup=None, wnr=None, dfac=None,
                 dep=None):
        super().__init__()
        self.names = [['noite', 'kdata', 'mcyc', 'msec', 'mcypr', 'mop', 'texp', 'be'],
                      ['tstart', 'timax', 'delten', 'deltmx', 'elst', 'gf', 'redlt', 'scale'],
                      ['dlt'],
                      ['re1', 're2', 'wup', 'wnr', 'dfac'],
                      ['dep']]
        self.mop = mop
        self.dlt = dlt
        self.dep = dep
        self.update_records(noite, kdata, mcyc, msec, mcypr, mop, texp, be,
                            tstart, timax, delten, deltmx, elst, gf, redlt, scale,
                            dlt,
                            re1, re2, wup, wnr, dfac,
                            dep)

    def update_records(self, *args, **kwargs):

        self.records = []
        if not args and not kwargs:
            args = []
            for rec_name in self.names:
                for name in rec_name:
                    if name is not None:
                        args.append(getattr(self, name))

        # PARAM.1:
        record = TOUGHRecord()

        # Add NOITE, KDATA, MCYC, MSEC, MCYPR to initial record
        for name, arg, data_fmt in zip(self.names[0], args, ['{:>2}']*2 + ['{:>4}']*3):
            record.append((name, arg, data_fmt))

        # Add MOP list to initial record
        for mop_i in args[5] if args[5] is not None else [0]*24:
            record.append((None, mop_i, '{:>1}'))

        # Add 10 blank places to initial record:
        record.append((None, None, '{:>10}'))

        for name, arg in zip(self.names[0][-2:], args[6:8]):
            record.append((name, arg, '{:>10.4E}'))

        # Add PARAM.1 to record list:
        self.append(record)

        # PARAM.2:
        record = TOUGHRecord()

        data_fmts = ['{:>10.4E}']*4 + ['{:>5}']*2 + ['{:>10.4E}']*3
        names = self.names[1]
        names.insert(5, None)
        args_list = list(args[8:16])
        args_list.insert(5, None)

        for name, arg, data_fmt in zip(names, args_list, data_fmts):
            record.append((name, arg, data_fmt))

        self.append(record)

        # PARAM.2.1,2.2, etc.:
        if args[10] is not None and args[10] < 0.0:
            self.add_tables(['dlt'], 8, '{:>10.4E}')

        # PARAM.3
        record = TOUGHRecord()
        for name, arg in zip(self.names[3], args[17:22]):
            record.append((name, arg, '{:>10.4E}'))
        self.append(record)

        # PARAM.4
        if getattr(self,'dep') is not None:
            self.add_tables(['dep'], 4, '{:>20.8E}')
        else:
            self.append(TOUGHRecord())

        return None

    @classmethod
    def empty(cls):
        mop = [0]*24
        return ParamCollection(noite=0, kdata=0, mcyc=0, msec=0, mcypr=0, mop=mop, texp=0.0, be=0.0,
                               tstart=0.0, timax=0.0, delten=0.0, deltmx=0.0, elst='NOELM', gf=0.0, redlt=0.0,scale=0.0,
                               re1=0.0, re2=0.0, wup=0.0, wnr=0.0, dfac=0.0)

    @classmethod
    def from_file(cls, data_record):
        # PARAM.1:
        pc, reject_list = super().from_file(data_record[0])
        pc.mop = reject_list[:24]

        # PARAM.2:
        pc.records[1].update_from_file(data_record[1])
        for entry in pc.records[1].entries:
            if entry.name is not None:
                setattr(pc, entry.name, entry.value)

        # PARAM.2.1, 2.2, etc.:
        delten = getattr(pc, 'delten')
        next_rec = 2
        if delten is not None and delten < 0.0:
            # Read in table of times values
            removed_records = pc.records[next_rec]
            pc.records = pc.records[:2]
            num_dlt = int(abs(delten))
            pc.dlt, data_record = pc.read_table(data_record[2:], num_dlt, 8, '{:>10.4E}')
            next_rec += int(np.ceil(num_dlt/8.0))
            pc.append(removed_records)

        # PARAM.3:
        pc.records[next_rec].update_from_file(data_record[0])
        for entry in pc.records[next_rec].entries:
            if entry.name is not None:
                setattr(pc, entry.name, entry.value)

        # PARAM.4:
        pc.dep, _ = pc.read_table(data_record[1:], None, 4, '{:>20.13E}')

        return pc, []


class Selec(TOUGHSimpleBlock):

    def __init__(self, selec_collection=None, ie=None, fe=None):
        super().__init__(record_collections=selec_collection, trc=SelecCollection)
        self.names = ['ie','fe']
        if selec_collection is None:
            self.fill_attributes(ie=ie, fe=fe)
            self.trc_from_args(ie, fe)

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False):
        return super().from_file(fn, trc=SelecCollection, names=['ie', 'fe'])


class SelecCollection(TOUGHRecordCollection):

    def __init__(self, ie=None, fe=None):
        super().__init__()
        self.names = [['ie'],
                      ['fe']]
        self.update_records(ie, fe)

    def update_records(self, *args, **kwargs):

        # if not args and not kwargs:
        self.records = []
        if args or kwargs:
            setattr(self, 'ie', args[0])
            setattr(self, 'fe', args[1])

        self.add_tables(['ie'], 16, '{:>5}')
        self.add_tables(['fe'], 8, '{:>10.4E}')

        return None

    @classmethod
    def empty(cls):
        return SelecCollection([0]*16, [0.0]*64*8)

    @classmethod
    def from_file(cls, data_record):
        selec_collection = SelecCollection.empty()
        selec_collection.ie, _ = selec_collection.read_table([data_record[0]], None, 16, '{:>5}', data_type='int')
        num_records = selec_collection.ie[0]
        if num_records is not None and num_records > 0:
            # Read in table of times values
            selec_collection.fe, lines = selec_collection.read_table(data_record[1:num_records+1],
                                                                     None, 8, '{:>10.4E}')
            return selec_collection, []
        else:
            return selec_collection, []


class Eleme(TOUGHBlock):
    pass


class Elements(TOUGHRecordCollection):
    pass


class Element(TOUGHRecord):
    pass


class Conne(TOUGHBlock):
    pass


class Connections(TOUGHRecordCollection):
    pass


class Connection(TOUGHRecord):
    pass


class Gener(TOUGHBlock):

    def __init__(self, gener_terms=None):
        super().__init__(record_collections=gener_terms, end_with_blank_line=True)

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=True, return_lines_indices=False):
        return super().from_file(fn, trc=GenerTerm, end_with_blank_line=end_with_blank_line,
                                 return_line_indices=return_lines_indices)


class GenerTerm(TOUGHRecordCollection):

    """ represents a term in the GENER block, with source/sink values and/or schedules """
    def __init__(self, element, code_name, gen_type, gx, ltab=None, nseq=None, nadd=None, nads=None, t_gen=None,
                 ex=None, hg=None):
        super().__init__()
        self.names = [['element', 'code_name', 'nseq', 'nadd', 'nads', 'ltab', 'type', 'itab', 'gx', 'ex', 'hg'],
                      ['t_gen'],
                      ['gx'],
                      ['ex']]
        args = [element, code_name, gen_type, gx]
        kwargs = {'ltab': ltab, 'nseq': nseq, 'nadd': nadd, 'nads': nads, 't_gen': t_gen, 'ex':ex, 'hg': hg}

        self.update_records(*args, **kwargs)

    @classmethod
    def empty(cls):
        return GenerTerm('NOELM', 'NOGNR', 'NTGN', 0.0)

    @classmethod
    def from_file(cls, lines, data_fmts=None):
        if type(lines) == list:
            gen_term, _ = super().from_file(lines[0])
            # Check if additional records are provided for this GENER term (time schedules of sources/sinks):
            ltab = getattr(gen_term, 'ltab')
            itab = getattr(gen_term, 'itab')
            if (ltab > 1) and (itab is not None):
                # Read in table of times and generation rates:
                gen_term.t_gen, lines = gen_term.read_table(lines[1:], ltab, 4, '{:>14.7E}')
                gen_term.gx, lines = gen_term.read_table(lines, ltab, 4, '{:>14.7E}')
                if itab is not None:
                    # Read in table of enthalpies (may be redundant now to make an if statement, 9/7/21)
                    gen_term.ex, lines = gen_term.read_table(lines, ltab, 4, '{:>14.7E}')
        else:
            gen_term = super().from_file(lines[0])
            lines = []

        return gen_term, lines

    def update_records(self, *args, **kwargs):

        args, kwargs = super().update_records(*args, **kwargs)

        self.t_gen = kwargs['t_gen']
        self.gx = args[3]
        self.ex = kwargs['ex']
        self.hg = kwargs['hg']
        if kwargs['ltab'] is None:
            ltab = 1 if self.t_gen is None else len(self.t_gen)
        else:
            ltab = kwargs['ltab']
        record = TOUGHRecord()
        record.append([('element', args[0], '{:>5}'),
                       ('code_name', args[1], '{:>5}'),
                       ('nseq', kwargs['nseq'], '{:>5}'),
                       ('nadd', kwargs['nadd'], '{:>5}'),
                       ('nads', kwargs['nads'], '{:>5}'),
                       ('ltab', ltab, '{:>5}'),
                       (None, None, '{:>5}'),
                       ('type', args[2], '{:>4}')])

        if self.t_gen is None:
            # Only a constant source/sink term is provided, not a schedule
            record.append([('itab', None, '{:>1}'),
                           ('gx', self.gx, '{:>10.3E}'),
                           ('ex', self.ex, '{:>10.4E}'),
                           ('hg', self.hg, '{:>10.4E}')])
            self.append(record)

        else:
            # A time schedule is provided for this source/sink
            itab = None
            if (self.ex is not None) and (type(self.ex) != float):
                # A schedule of enthalpies is also provided
                itab = 'E'
            record.append([('itab', itab, '{:>1}'),
                           (None, None, '{:>10.4E}'),
                           ('ex', self.ex, '{:>10.4E}') if type(self.ex) != list else (None, None, '{:>10.4E}'),
                           ('hg', self.hg, '{:>10.4E}')])
            self.append(record)
            table_names = ['t_gen', 'gx', 'ex'] if itab is not None else ['t_gen', 'gx']
            self.add_tables(table_names, 4, '{:>14.7E}')

        return None

    def name_to_args(self, args, kwargs):

        args_in = list(args)
        args = tuple(args_in[:2] + [args_in[6], args_in[8]])
        kwargs = {}
        for name in ['ltab', 'nseq', 'nadd', 'nads', 't_gen', 'ex', 'hg']:
            kwargs.update({name: getattr(self, name)})

        return args, kwargs


class Sink(GenerTerm):

    """Represents a TOUGH Sink term, which is a special type of GenerTerm class"""
    def __init__(self, element, code_name, gx, ltab=None, nseq=None, nadd=None, nads=None, t_gen=None, ex=None,
                 hg=None):
        super(Sink, self).__init__(element, code_name, 'MASS', -1.0*np.absolute(gx),
                                   ltab=ltab, nseq=nseq, nadd=nadd, nads=nads, t_gen=t_gen, ex=ex, hg=hg)


class Diffu(TOUGHSimpleBlock):

    def __init__(self, diffu_collection=None, fddiag=None):

        super().__init__(record_collections=diffu_collection, trc=DiffuCollection)
        if diffu_collection:
            self.fill_attributes(fddiag=diffu_collection[0].fddiag)
            self.trc_from_args(diffu_collection[0].fddiag)

    @classmethod
    def from_file(cls, fn, names=None, trc=None, end_with_blank_line=False):
        return super().from_file(fn, trc=DiffuCollection, names=['fddiag'])


class DiffuCollection(TOUGHRecordCollection):

    def __init__(self, fddiag=None):
        super().__init__()
        self.names = ['fddiag']
        self.update_records(fddiag)

    def update_records(self, *args, **kwargs):

        self.records = []
        if args or kwargs:
            setattr(self, 'fddiag', args[0])

        for diff_k in getattr(self, 'fddiag'):
            rec = TOUGHRecord()
            for diff_kph in diff_k:
                rec.append((None, diff_kph, '{:>10.4E}'))
            self.append(rec)

        return None

    @classmethod
    def empty(cls):
        return DiffuCollection([[0.0,0.0]])

    @classmethod
    def from_file(cls, data_record):
        diffu_collection = DiffuCollection.empty()
        diffu_collection.fddiag = []

        for dr in data_record:
            fd, _ = diffu_collection.read_table([dr], None, 8, '{:>10.4E}')
            diffu_collection.fddiag.append(fd.tolist())

        return diffu_collection, []


class Incon(TOUGHBlock):

    def __init__(self, incons=None, footers=None):
        super().__init__(record_collections=incons, end_with_blank_line=True)
        if footers is None:
            self.footers = [' ']
        else:
            self.footers = footers

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=True, extra_record=False, return_line_indices=False):
        print('Finding lines in file:')
        if return_line_indices:
            lines, footers, ind_lines = cls.find_lines(fn, end_with_blank_line=end_with_blank_line, return_indices=True)
        else:
            ind_lines = None
            lines, footers = cls.find_lines(fn, end_with_blank_line=end_with_blank_line, return_indices=False)

        block = cls.block_from_lines(lines, trc=InconCollection, extra_record=extra_record)
        block.footers = footers
        if return_line_indices:
            return block, ind_lines
        else:
            return block

    @classmethod
    def find_lines(cls, fn, end_with_blank_line=False, return_indices=False):
        # This method reads in all records from a TOUGH Block and pulls in all text lines need to generate
        # a TOUGHBlock object.
        f = open(fn, 'r')
        lines_list = f.readlines()
        return cls.find_lines_from_list(lines_list, end_with_blank_line, return_indices)

    @classmethod
    def find_lines_from_list(cls, lines_list, end_with_blank_line=False, return_indices=False):
        # This method reads in all records from a TOUGH Block and pulls in all text lines need to generate
        # a TOUGHBlock object.
        keyword = cls.__name__[:5].upper()
        eligible_keywords = cls.get_eligible_keywords()
        lines = []
        footers = []
        ind_lines = []
        read_footer = False
        read_lines = False
        for i_line, line in lines_list:
            if line[:len(keyword)] == keyword:
                read_lines = True
                ind_lines.append(i_line)
                continue
            if read_lines:
                if read_footer:
                    if end_with_blank_line and not line.strip():
                        break
                    else:
                        ind_lines.append(line)
                        footers.append(line)
                else:
                    if line[:5] in eligible_keywords:
                        break
                    elif end_with_blank_line and not line.strip():
                        break
                    elif line[:3] == '+++':
                        ind_lines.append(i_line)
                        footers.append(line)
                        read_footer = True
                    else:
                        ind_lines.append(line)
                        lines.append(line)

        if return_indices:
            return lines, footers, ind_lines
        else:
            return lines, footers

    @classmethod
    def block_from_lines(cls, lines, trc=None, extra_record=False):

        record_collections = []
        num_lines = 2 + int(extra_record)
        num_incons = int(len(lines)/num_lines)

        for i_inc in np.arange(num_incons):
        # while len(lines) > 0:
            # Parse lines for data for each record collection. Return a record collection object and the remaining
            # lines to be read.
            # num_lines = 2 + int(extra_record)
            # print(lines[:num_lines])
            # exit()
            rc, _ = InconCollection.from_file(lines[num_lines*i_inc:num_lines*(i_inc+1)], extra_record=extra_record)
            # lines = lines[num_lines:]
            # if len(lines) % 1000 == 0:
                # print(str(len(lines)) + ' lines left.')
            if i_inc % 10000 == 0:
                print('On number ' + str(i_inc) + ' of ' + str(num_incons))
            record_collections.append(rc)

        return cls(record_collections)

    def to_file(self, fname=None, update_records=True, prepend_line=None):
        block_str = self.keyword + '----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
        if type(self.record_collections) == list:
            rcs = self.record_collections
        else:
            rcs = [self.record_collections]
        for rc in rcs:
            block_str += rc.to_file(update_records=update_records)
        block_str += '\n'
        for footer in self.footers:
            block_str += footer

        if fname is not None:
            f = open(fname, 'w')
            f.write(block_str)
            f.close()

        return block_str


class InconCollection(TOUGHRecordCollection):

    def __init__(self, element, x, nseq=None, nadd=None, porx=None):
        super().__init__()
        self.names = [['element', 'nseq', 'nadd', 'porx'],
                      ['x']]
        args = [element, x]
        setattr(self,'x', x)
        kwargs = {}
        kwargs.update({'nseq': nseq})
        kwargs.update({'nadd': nadd})
        kwargs.update({'porx': porx})

        self.update_records(*args,**kwargs)

    def update_records(self, *args, **kwargs):

        if not args and not kwargs:
            kwargs = {}
            args = [getattr(self,self.names[0][0]), getattr(self,'x')]
            args = tuple(args)
            if type(self).__name__ == 'InconCollection':
                kwargs.update({'nseq':getattr(self, 'nseq')})
                kwargs.update({'nadd':getattr(self, 'nadd')})
                kwargs.update({'porx':getattr(self, 'porx')})

        self.records = []

        # INCON.1:
        record = TOUGHRecord()
        record.append((self.names[0][0], args[0], '{:>5}'))
        if type(self).__name__ == 'InconCollection':
            record.append(('nseq', kwargs['nseq'], '{:>5}'))
            record.append(('nadd', kwargs['nadd'], '{:>5}'))
            record.append(('porx', kwargs['porx'], '{:>15.9E}'))
        self.append(record)

        # INCON.2:
        self.add_tables(['x'], 4, '{:>20.13E}')

        return None

    @classmethod
    def empty(cls):
        return InconCollection('NOELM', [0.0], 0, 0, 0.0)

    @classmethod
    def from_file(cls, data_record, extra_record=False):

        block, _ = super().from_file(data_record[0])
        final_record = 1 + int(extra_record)
        x, _ = block.read_table(data_record[1:final_record+1], None, 4, '{:>20.13E}')
        setattr(block, 'x', x)

        return block, data_record[final_record:]


class Indom(Incon):

    def __init__(self, indoms=None):
        Incon.__init__(self, incons=indoms)

    @classmethod
    def from_file(cls, fn, trc=None, end_with_blank_line=True, extra_record=False):
        lines, footers = cls.find_lines(fn, end_with_blank_line=end_with_blank_line)
        block = cls.block_from_lines(lines, trc=IndomCollection, extra_record=extra_record)
        block.footers = footers
        return block


class IndomCollection(InconCollection):

    def __init__(self, mat, x):
        InconCollection.__init__(self, mat, x)
        self.names[0][0] = 'mat'
        setattr(self, 'mat', getattr(self, 'element'))
        delattr(self, 'element')

    @classmethod
    def from_file(cls, data_record, extra_record=False):
        block = super().from_file(data_record, extra_record=extra_record)

        return block

    @classmethod
    def empty(cls):
        return IndomCollection('NOMAT', [0.0])


if __name__ == '__main__':

    from os.path import dirname as up

    # base_dir = os.path.join(os.pardir, 'test_data')
    base_dir = os.path.join(up(up(up(os.getcwd()))), 'output')
    fname = os.path.join(base_dir, 'flow_chk.inp')
    param = Param.from_file(fname)
    exit()
    # rocks, i_lines = Momop.from_file(fname, return_line_indices=True)
    tough_input = TOUGHInput.from_file(fname)
    print(tough_input.keyword_list)
    exit()
    fname_out = os.path.join(base_dir, 'INFILE_chk')

    incon = Incon.from_file(fname, extra_record=True)
    print(incon.to_file())
    exit()

    react = React.from_file(fname)
    react = Momop(mop2_list=[1,2,None,3])
    print(react.to_file())

    tough_inp = TOUGHInput(title='Testing this preprocessor routine')
    rocks = Rocks.from_file(fname)
    delattr(rocks[0], 'irp')
    delattr(rocks[0], 'rp')
    delattr(rocks[0], 'icp')
    delattr(rocks[0], 'cp')
    setattr(rocks[0], 'nad', 1)
    # tough_inp.append(rocks)
    print(rocks.to_file(update_records=False))
    gener = Gener.from_file(fname)
    # tough_inp.append(gener)
    # gener_term = GenerTerm('AAA 1', 'SRC 1', 'COM4', np.array([1.0, 2.0, 3.0, 4.0]),
    #                        t_gen=np.array([1.0, 2.0, 3.0, 4.0]), ex=[2.0, 4.0, 6.0, 8.0])
    # gener = Gener()
    # gener.append(gener_term)
    # print(gener[0].t_gen)
    # print(gener[0].gx)
    # print(gener[0].gx)
    # print(gener.to_file())

    selec = Selec.from_file(fname)
    tough_inp.append(selec)
    # print(selec.to_file())
    # rpcap = RpCap(rpcp=RpCp(11, [0.5,0.0,0.0], 11, [1.67, 1.8e7, 1.0e9]))
    rpcap = RpCap.from_file(fname)
    tough_inp.append(rpcap)
    # print(rpcap.to_file())
    multi = Multi.from_file(fname)
    tough_inp.append(multi)
    # print(multi.to_file())
    times = Times.from_file(fname)
    tough_inp.append(times)
    # print(times.to_file())
    start = Start()
    tough_inp.append(start)
    param = Param.from_file(fname)
    tough_inp.append(param)
    # print(param.to_file())
    diffu = Diffu.from_file(fname)
    tough_inp.append(diffu)
    tough_inp.append(gener)
    # print(diffu.to_file())
    # incon = Incon.from_file(fname, extra_record=True)
    # print(incon.to_file())
    # indom = Indom()
    # indom.append(IndomCollection('MAT 1', [1.0e5,0.0,0.0,0.0,0.0,40.0]))
    # indom.append(IndomCollection('MAT 2', [1.0e6,1.0,0.0,0.0,0.0,25.0]))
    # print(indom.to_file())
    endcy = Endcy()
    tough_inp.append(endcy)
    # print(endcy.to_file())
    tough_inp.to_file(fname_out)

# def float_or_none(s):
#     """ converts a string to a float; if it is empty, it will return None"""
#     try:
#         return float(s)
#     except ValueError:
#         return None
#
# class Eleme():
#     """ represents an Eleme record.  See p172 of the TOUGH2 manual for what the
#     properties mean.  """
#     def __init__(self, name, nseq, nadd, ma1, ma2, volx, ahtx, pmx, x, y, z):
#         self.name = name
#         self.nseq = nseq
#         self.nadd = nadd
#         self.ma1 = ma1
#         self.ma2 = ma2
#         self.volx = volx
#         self.ahtx = ahtx
#         self.pmx = pmx
#         self.x = x
#         self.y = y
#         self.z = z
#         self.connections = []         # List of connection indices of which this element is a part
#         self.is_n1 = []               # List of booleans indicating if Eleme is n1 in each of self.connections
#         self.connected_elements = []  # List of connected elements
#
#     def as_numpy_array(self):
#         # Generate numpy arrays of ELEME
#         dt = np.dtype([('name', 'U5'),
#                        ('nseq', 'U5'),
#                        ('nadd', 'U5'),
#                        ('ma1', 'U3'),
#                        ('ma2', 'U2'),
#                        ('volx', np.float64),
#                        ('ahtx', np.float64),
#                        ('pmx', np.float64),
#                        ('x', np.float64),
#                        ('y', np.float64),
#                        ('z', np.float64), ])
#
#         data_eleme = np.empty(1, dtype=dt)
#
#         data_eleme['name'] = self.name
#         data_eleme['nseq'] = self.nseq
#         data_eleme['nadd'] = self.nadd
#         data_eleme['ma1'] = self.ma1
#         data_eleme['ma2'] = self.ma2
#         data_eleme['volx'] = self.volx
#         data_eleme['pmx'] = 1.0 if self.pmx is None else self.pmx
#         data_eleme['ahtx'] = 0.0 if self.ahtx is None else self.ahtx
#         data_eleme['x'] = self.x
#         data_eleme['y'] = self.y
#         data_eleme['z'] = self.z
#
#         return data_eleme
#
# class ElemeCollection():
#     """ represents an ordered set of Elements, as read from the mesh file """
#     def __init__(self, fname):
#         self.fname = fname
#         self.elements = []
#         self.name2idx = dict()
#
#     def proc_nodes(self, ):
#         for idx, node in enumerate(gen_nodes(self.fname)):
#             self.elements.append(node)
#             self.name2idx[node.name] = idx
#
#     def __getitem__(self, item):
#         if type(item) == int:
#             # item is element listing index
#             return self.elements[item]
#         elif type(item) == str:
#             # item is an element name
#             return self.elements[self.name2idx[item]]
#         elif type(item) == list:
#             return_list = []
#             for i_item in item:
#                 return_list.append(self[i_item])
#             return return_list
#
#     def __len__(self):
#         return len(self.elements)
#
#     def as_numpy_array(self):
#         # Generate numpy arrays of ELEME
#         dt = np.dtype([('name', 'U5'),
#                        ('nseq', 'U5'),
#                        ('nadd', 'U5'),
#                        ('ma1', 'U3'),
#                        ('ma2', 'U2'),
#                        ('volx', np.float64),
#                        ('ahtx', np.float64),
#                        ('pmx', np.float64),
#                        ('x', np.float64),
#                        ('y', np.float64),
#                        ('z', np.float64), ])
#
#         data_eleme = np.empty(len(self.elements), dtype=dt)
#
#         for i_el, elem in enumerate(self.elements):
#             # data_eleme[i_el] = elem.as_numpy_array()
#             data_eleme['name'][i_el] = elem.name
#             data_eleme['nseq'][i_el] = elem.nseq
#             data_eleme['nadd'][i_el] = elem.nadd
#             data_eleme['ma1'][i_el] = elem.ma1
#             data_eleme['ma2'][i_el] = elem.ma2
#             data_eleme['volx'][i_el] = elem.volx
#             data_eleme['pmx'][i_el] = 1.0 if elem.pmx is None else elem.pmx
#             data_eleme['ahtx'][i_el] = 0.0 if elem.ahtx is None else elem.ahtx
#             data_eleme['x'][i_el] = elem.x
#             data_eleme['y'][i_el] = elem.y
#             data_eleme['z'][i_el] = elem.z
#
#         return data_eleme
#
#     def update_from_numpy_array(self, data_eleme, col_names = None, idx = None):
#
#         if idx is None:
#             # Update all elements:
#             idx = np.arange(len(self.elements))
#         if col_names is None:
#             # Update all columns
#             col_names = ['name', 'nseq', 'nadd', 'ma1', 'ma2', 'volx', 'pmx', 'ahtx', 'x', 'y', 'z']
#
#         if len(data_eleme) != len(self):
#             # At least one element has been added or removed, so regenerate new list of Eleme objects
#             print('The length of the numpy array data_eleme does not match the number of elements in the mesh'
#                   'object.')
#             exit()
#
#         else:
#             # No elements were added or removed.  Only the attributes listed in col_names have been changed
#             for i_el in idx:
#                 for col_name in col_names:
#                     # Update only columns provided in col_names
#                     setattr(self.elements[i_el], col_name, data_eleme[col_name][i_el])
#
#         return None
#
#     def change_ma_of_elements(self, elem_list, to_ma):
#
#         # Changes materials type of all elements whose indices are provided in elem_list to material type to_ma
#         eleme_data = self.as_numpy_array()
#
#         eleme_data['ma1'][elem_list] = to_ma[:-2]
#         eleme_data['ma2'][elem_list] = to_ma[-2:]
#         self.update_from_numpy_array(eleme_data, col_names=['ma1', 'ma2'], idx=elem_list)
#
#         return None
#
#     def to_file(self, f):
#         # Print ELEME list to file 'f' (either filename or handle)
#         hdr = 'ELEME----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
#         data_eleme = self.as_numpy_array()
#         fmt = ['%5s', '%5s', '%5s', '%3s', '%2s', '%10.4E', '%10.4E','%10.4E','%10.4E','%10.4E','%10.4E']
#         np.savetxt(f, data_eleme, header=hdr, delimiter='', fmt=fmt, comments='')
#
#         return None
#
#     def displace(self, delta_x, delta_y, delta_z):
#
#         for elem in self.elements:
#             elem.x += delta_x
#             elem.y += delta_y
#             elem.z += delta_z
#
#         return None
#
# class Incon():
#
#     def __init__(self, element, X, nseq=None, nadd=None, porx=None):
#         self.element = element
#         self.X = X
#         self.nseq = nseq
#         self.nadd = nadd
#         self.porx = porx
#
#     @property
#     def num_X(self):
#         return len(self.X)
#
#     def to_file(self, f, append=True):
#
#         incon_dat = '{:>5}'.format(self.element)
#         incon_dat += ' '*5 if self.nseq is None else '{:>5}'.format(self.nseq)
#         incon_dat += ' '*5 if self.nadd is None else '{:>5}'.format(self.nadd)
#         incon_dat += ' ' *15 if self.porx is None else '{:>15.9E}'.format(self.porx)
#         incon_dat += '\n'
#
#         for x in self.X:
#             incon_dat += ' {:>19.13E}'.format(x)
#
#         if not append:
#             f = open(f,'w')
#
#         f.write(incon_dat)
#
#         if not append:
#             f.close()
#
#         return None
#
#
# class InconCollection():
#
#     def __init__(self):
#         self.incons = []
#         self.footer = None
#
#     def append(self, incon):
#         self.incons.append(incon)
#         return None
#
#     def to_file(self, fname):
#
#         f = open(fname, 'w')
#         hdr_dat = 'INCON----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
#         f.write(hdr_dat)
#         for incon in self.incons:
#             f.write('\n')
#             incon.to_file(f)
#         f.write('\n')
#
#         if self.footer is not None:
#             f.write('\n')
#             f.write(self.footer)
#             f.write('\n')
#
#         f.close()
#
#         return None
#
#
# class Gener():
#
#     """ represents an entry into the GENER block, with source/sink terms """
#     def __init__(self, element, code_name, gen_type, gx, nseq=None, nadd=None, nads=None,
#                  t_gen=None, ex=None, hg=None):
#         self.element = element
#         self.code_name = code_name
#         self.nseq = nseq
#         self.nadd = nadd
#         self.nads = nads
#         self.type = gen_type
#         self.t_gen = t_gen
#         self.gx = gx
#         self.ex = ex
#         self.hg = hg
#
#     @property
#     def ltab(self):
#         if type(self.gx) is float:
#             return 1
#         else:
#             return len(self.gx)
#
#     @property
#     def itab(self):
#         if self.ex is None:
#             return 0
#         elif type(self.ex) is float:
#             return 1
#         else:
#             return len(self.ex)
#
#     def to_file(self, f, append=True):
#
#         hdr_dat = '{:>5}{:>5}'.format(self.element, self.code_name)
#         hdr_dat += ' '*5 if self.nseq is None else '{:>5}'.format(self.nseq)
#         hdr_dat += ' '*5 if self.nadd is None else '{:>5}'.format(self.nadd)
#         hdr_dat += ' '*5 if self.nads is None else '{:>5}'.format(self.nads)
#         hdr_dat += '{:>5}'.format(self.ltab)
#         hdr_dat += ' '*5
#         hdr_dat += '{:>4}'.format(self.type)
#         hdr_dat += 'E' if self.itab > 1 else ' '
#         if self.ltab > 1:
#             hdr_dat += ' '*10
#         else:
#             hdr_dat += '{:>10.4E}'.format(self.gx) if self.gx >= 0.0 else '{: 9.3E}'.format(self.gx)
#         hdr_dat += ' '*10 if (self.itab > 1 or self.ex is None) else '{:>10.4E}'.format(self.ex)
#         hdr_dat += ' '*10 if self.hg is None else '{:>10.4E}'.format(self.hg)
#
#         if not append:
#             # Writing a new file, in which case 'f' is a file handle
#             f = open(f, 'w')
#         f.write(hdr_dat)
#
#         if self.ltab > 1:
#             n_lines = int(np.ceil(self.ltab/4))
#             tab_data = [self.t_gen, self.gx, self.ex] if self.itab > 1 else [self.t_gen, self.gx]
#             for dat in tab_data:
#                 n_written = 0
#                 for _ in np.arange(n_lines):
#                     f.write('\n')
#                     line_dat = ''
#                     n_write = np.min([self.ltab-n_written, 4])
#                     for entry in np.arange(n_write):
#                         line_dat += '{:>14.7E}'.format(dat[n_written + entry])
#                     f.write(line_dat)
#                     n_written += n_write
#
#         return None
#
# class GenerCollection():
#     """Represents an entire GENER block"""
#     def __init__(self, mop12, ):
#         self.mop12 = mop12
#         self.geners = []
#
#     def __len__(self):
#         return len(self.geners)
#
#     def __getitem__(self, item):
#         return self.geners[item]
#
#     def append(self, gener):
#         self.geners.append(gener)
#         return
#
#     def to_file(self, fname):
#
#         hdr_dat = 'GENER----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
#         f = open(fname, 'w')
#         f.write(hdr_dat)
#
#         for gener in self.geners:
#             f.write('\n')
#             gener.to_file(f)
#
#         f.write('\n')
#         f.close()
#
#         return None
#
# class Mesh():
#     """ represents a mesh; nodes and their connections """
#     def __init__(self, fname):
#         self.nodes = ElemeCollection(fname)
#         self.nodes.proc_nodes()
#         self.connections = ConneCollection(fname)
#         self.connections.proc_conne()
#         self.proc_nodes()
#         self._points = None
#
#     def connected_elements(self, node):
#         """ given a node name (str), return
#         the connected element index """
#
#         con_els = []  # Initializing list of connected elements
#         i_con_els = []
#         is_n1 = []
#
#         for icon in self.connections.connections_for_node[node]:
#             con = self.connections[icon]
#             n1 = con.name1
#             n2 = con.name2
#             if ((node != n1) and (self.nodes.name2idx[n1] not in con_els)):
#                 # node is n2
#                 is_n1.append(False)
#                 con_els.append(self.nodes.name2idx[n1])
#                 i_con_els.append(self.nodes.name2idx[n1])
#             elif ((node != n2) and (self.nodes.name2idx[n2] not in con_els)):
#                 # note is n1
#                 is_n1.append(True)
#                 con_els.append(self.nodes.name2idx[n2])
#                 i_con_els.append(self.nodes.name2idx[n2])
#
#         isrt = np.argsort(i_con_els) # Arranges lists in order that connected nodes appear in ELEME list
#
#         return np.array(con_els)[isrt].tolist(), np.array(is_n1)[isrt].tolist(), isrt
#
#     def proc_nodes(self):
#         # The Mesh 'proc_nodes' routine populates the connected_elements list for each Eleme object
#         for elem in self.nodes.elements:
#             elem.connected_elements, elem.is_n1, isrt = self.connected_elements(elem.name)
#             elem.connections = np.array(self.connections.connections_for_node[elem.name])[isrt].tolist()
#
#     @property
#     def points(self):
#         """ return a numpy array of [x, y, z, vol, ma1, ma2]
#         points ordered by node order """
#         if self._points is None:
#             n = len(self.nodes)
#             self._points = np.zeros([n,4])
#             for ix, n in enumerate(self.nodes):
#                 self._points[ix,:] = [n.x, n.y, n.z, n.volx]#, n.ma1, n.ma2]
#             return self.points
#         return self._points
#
#     def as_data_frame(self):
#         """ return a pandas dataframe.  Index is the node name,
#         has columns x, y, z, and volx"""
#         #data = self.points
#         names = [i.name for i in self.nodes]
#
#         df = pd.DataFrame.from_records(
#             data=([n.x, n.y, n.z, n.volx, n.ma1, n.ma2] for n in self.nodes),
#             index=names, columns=["x","y","z","volx", "ma1","ma2"])
#
#         df['ma'] = df['ma1']+df['ma2']
#         keys = df['ma'].unique()
#         z = lambda x: float(np.where(keys == x)[0][0])
#         df['ma_code'] = df['ma'].map(z)
#         return df
#
#     def remove_nodes_of_type(self, ma):
#         # This routine finds all elements with materials type 'ma' (a string), removes from the ELEME block,
#         # and removes their instances in the CONNE block.
#
#         tmp_els = []
#         tmp_cons = []
#         for elem in self.nodes.elements:
#             if elem.ma1 + elem.ma2 != ma:
#                 tmp_els.append(elem)
#
#
#         for conn in self.connections.connections:
#
#             ma_1 = (self.nodes.elements[self.nodes.name2idx[conn.name1]].ma1 +
#                     self.nodes.elements[self.nodes.name2idx[conn.name1]].ma2)
#             ma_2 = (self.nodes.elements[self.nodes.name2idx[conn.name2]].ma1 +
#                     self.nodes.elements[self.nodes.name2idx[conn.name2]].ma2)
#             if ma_1 != ma and ma_2 != ma:
#                 tmp_cons.append(conn)
#
#         self.nodes.elements = tmp_els
#         # self.nodes.proc_nodes()
#         self.connections.connection = tmp_cons
#         # self.connections.proc_conne()
#         # self.proc_nodes()
#
#         return None
#
#     def replace_nodes_of_type(self, from_ma, to_ma, bound_el=False, bound_el_name = None, new_vol = None, d12 = None,
#                               new_ahtx = 0.0, new_pmx = 1.0, new_x = 0.0, new_y = 0.0, new_z = 0.0,
#                               elem_data=None, iex_rm=None, new_els=None, update_elements=True,
#                               conn_data=None, icx_rm=None, update_connections=True):
#
#         # Changes material type of all nodes having type "from_ma" to "to_ma".  If bound_el flag is set,
#         # all of these elements are replaced by a single boundary element having name provided by the user
#         # (bound_el_name), which must follow the TOUGH convention for element names (3 character string followed
#         # by two integers).  The volume of the element will be changed to variable new_vol if provided by the user.
#         # Otherwise, it will be the cumulative volume of all elements being replaced.  Likewise, the node distances
#         # d1 and d2 in the connections list will change to variable d12 if provided by the user.  Otherwise, these
#         # values will remain unchanged.
#
#         if elem_data is None:
#             # Generate a fresh numpy array of Eleme data:
#             elem_data = self.nodes.as_numpy_array()
#
#         elem_list = np.where(np.logical_and(elem_data['ma1'] == from_ma[:-2], elem_data['ma2'] == from_ma[-2:]))[0]
#         el_list = elem_data['name'][elem_list]
#
#         if from_ma != to_ma:
#             elem_data['ma1'][elem_list] = to_ma[:-2]
#             elem_data['ma2'][elem_list] = to_ma[-2:]
#             self.nodes.update_from_numpy_array(elem_data, col_names=['ma1', 'ma2'], idx=elem_list)
#
#         if bound_el:
#             if iex_rm is None:
#                 iex_rm = elem_list
#             else:
#                 iex_rm = np.unique(np.append(iex_rm, elem_list))
#             # All the elements of material type to_ma will be replaced with a single boundary element
#             volx = np.sum(elem_data['volx'][elem_list])
#             if update_elements:
#                 # Delete elements provided in iex_rm index list from list of Eleme objects and numpy array
#                 self.nodes.elements = np.delete(np.array(self.nodes.elements), iex_rm).tolist()
#                 elem_data = np.delete(elem_data, iex_rm)
#             # Generate a new boundary element (Eleme object):
#             new_el = Eleme(bound_el_name, '', '', to_ma[:-2], to_ma[-2:], volx if new_vol == None else new_vol,
#                            new_ahtx, new_pmx, new_x, new_y, new_z)
#             elem_data = np.append(elem_data, new_el.as_numpy_array())
#             if new_els is None:
#                 # Start list of new Eleme objects to append to Eleme object list
#                 new_els = [new_el]
#             else:
#                 # Add to list of new Eleme objects to append to Eleme object list
#                 new_els.append(new_el)
#             # Add new element object to Mesh:
#             if update_elements:
#                 # Add new Eleme objects to Eleme object list:
#                 self.nodes.elements += new_els
#             if conn_data is None:
#                 # Generate a fresh numpy array of ConneCollection data:
#                 conn_data = self.connections.as_numpy_array()
#             # Find connections having n1 as a boundary element:
#             idx_1 = np.nonzero(np.isin(conn_data['name1'], el_list))[0]
#             # Find connections having n2 as a boundary element:
#             idx_2 = np.nonzero(np.isin(conn_data['name2'], el_list))[0]
#             # Find connections having both n1 and n2 as a boundary element
#             idx_12, i_rm_1, i_rm_2 = np.intersect1d(idx_1, idx_2, assume_unique=True, return_indices=True)
#
#             if icx_rm is None:
#                 icx_rm = idx_12.tolist()
#             else:
#                 icx_rm = np.unique(np.append(icx_rm, idx_12)).tolist()
#
#             # Then remove those indices from the list of connections to be modified
#             idx_1 = np.delete(idx_1, i_rm_1)
#             idx_2 = np.delete(idx_2, i_rm_2)
#
#             # Modify element names from entries of CONNE list:
#             conn_data['name1'][idx_1] = bound_el_name
#             conn_data['name2'][idx_2] = bound_el_name
#
#             # Update attributes of Connection objects:
#             if d12 is None:
#                 # Only update element name in CONNE entry:
#                 self.connections.update_from_numpy_array(conn_data, col_names=['name1'], idx=idx_1)
#                 self.connections.update_from_numpy_array(conn_data, col_names=['name2'], idx=idx_2)
#             else:
#                 # Update both element name and d1/d2 in CONNE entry:
#                 conn_data['d1'][idx_1] = d12
#                 conn_data['d2'][idx_2] = d12
#                 self.connections.update_from_numpy_array(conn_data, col_names=['name1', 'd1'], idx=idx_1)
#                 self.connections.update_from_numpy_array(conn_data, col_names=['name2', 'd2'], idx=idx_2)
#
#             # Remove connection entries connecting two boundary elements from Conne object list:
#             if update_connections:
#                 # Connections with two boundary elements will be removed from the conn_data numpy array
#                 # and ConneCollection object:
#                 conn_data = np.delete(conn_data, icx_rm)
#                 self.connections.connections = np.delete(np.array(self.connections.connections), icx_rm).tolist()
#                 # Reset list of connections to remove:
#                 icx_rm = None
#
#         return elem_data, new_els, iex_rm, conn_data, icx_rm
#
#
#     def to_file(self, fname='MESH'):
#
#         # Writes Mesh object as ELEME and CONNE blocks to file fname
#
#         f = open(fname, 'w')
#         self.nodes.to_file(f)
#         f.write('\n')
#         self.connections.to_file(f)
#         f.write('\n')
#         f.close()
#
#         return None
#
#     @classmethod
#     def from_pickle(cls, file):
#         """ return a mesh from a pickled file """
#         with open(file, "rb") as f:
#             return pickle.load(f)
#
#     def to_pickle(self, file):
#         """ dump to pickle """
#         with open(file, 'wb') as f:
#             pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
#
# def gen_nodes(fname):
#     """ iterate through f and yield a node for each item """
#     with open(fname, 'r') as f:
#         idx = None
#         for line in f:
#             if "ELEME" in line:
#                 idx = 0
#                 continue
#             if idx is not None:
#                 if line.strip() == "":
#                     break
#                 name = line[0:5]
#                 """ five character code name of element"""
#                 nseq = line[5:10]
#                 """ number of additional elements having the same volume """
#                 nadd = line[10:15]
#                 """ increment between hte code numbers of two successsive elements """
#                 ma1 = line[15:18]
#                 """reserved code prefix """
#                 ma2 = line[18:20]
#                 """ reserved code suffux"""
#                 volx = float_or_none(line[20:30])
#                 """ element volume [m3] """
#                 ahtx = float_or_none(line[30:40])
#                 """ interface area [m2] """
#                 pmx = float_or_none(line[40:50])
#                 """ permeability modifier """
#                 x = float_or_none(line[50:60])
#                 y = float_or_none(line[60:70])
#                 z = float_or_none(line[70:80])
#                 """ cartesian coordinates of grid block centers.  """
#                 yield Eleme(name, nseq, nadd, ma1, ma2, volx, ahtx, pmx, x, y, z)
#                 idx+=1
#
# class Conne():
#     """ introduces information for the connections (interfaces) between elements
#     see appendix E, p173 of the TOUGH2 manual """
#
#     def __init__(self, name1, name2, nseq, nad1, nad2, isot, d1, d2, areax, betax, sigx):
#         self.name1 = name1
#         self.name2 = name2
#         self.nseq = nseq
#         self.nad1 = nad1
#         self.nad2 = nad2
#         self.isot = isot
#         self.d1 = d1
#         self.d2 = d2
#         self.areax = areax
#         self.betax = betax
#         self.sigx = sigx
#
# class ConneCollection():
#     """ interface to a collection of connection data """
#     def __init__(self, filename):
#         self.filename = filename
#         self.connections_for_node = dict()
#         self.connections = []
#
#     def proc_conne(self):
#         """ iterate the connections file and get """
#         for idx, conn in enumerate(gen_connections(self.filename)):
#             self.connections.append(conn)
#             n1 = conn.name1
#             n2 = conn.name2
#             for node in [n1, n2]:
#                 try:
#                     if idx not in self.connections_for_node[node]:
#                         self.connections_for_node[node].append(idx)
#                 except Exception:
#                     self.connections_for_node[node] = [idx]
#
#     def __getitem__(self, item):
#         if type(item) == int:
#             return self.connections[item]
#         if type(item) == str:
#             return [self.connections[i] for i in self.connections_for_node[item]]
#         if type(item) == Eleme:
#             return [self.connections[i] for i in self.connections_for_node[item.name]]
#
#     def __len__(self):
#         return len(self.connections)
#
#     def as_numpy_array(self):
#         # Generate numpy array of CONNE data
#         dt = np.dtype([('name1', 'U5'),
#                        ('name2', 'U5'),
#                        ('nseq', 'U5'),
#                        ('nad1', 'U5'),
#                        ('nad2', 'U5'),
#                        ('isot', np.int32),
#                        ('d1', np.float64),
#                        ('d2', np.float64),
#                        ('areax', np.float64),
#                        ('betax', np.float64),
#                        ('sigx', np.float64), ])
#         data_conne = np.empty(len(self.connections), dtype=dt)
#
#         for i_conn, conn in enumerate(self.connections):
#             data_conne['name1'][i_conn] = conn.name1
#             data_conne['name2'][i_conn] = conn.name2
#             data_conne['nseq'][i_conn] = conn.nseq
#             data_conne['nad1'][i_conn] = conn.nad1
#             data_conne['nad2'][i_conn] = conn.nad2
#             data_conne['isot'][i_conn] = conn.isot
#             data_conne['d1'][i_conn] = conn.d1
#             data_conne['d2'][i_conn] = conn.d2
#             data_conne['areax'][i_conn] = conn.areax
#             data_conne['betax'][i_conn] = 0.0 if conn.betax == None else conn.betax
#             data_conne['sigx'][i_conn] = 0.0 if conn.sigx  == None else conn.sigx
#
#         return data_conne
#
#     def update_from_numpy_array(self, data_conne, col_names=None, idx=None):
#         if idx is None:
#             # Update all connections:
#             idx = np.arange(len(self.connections))
#         if col_names is None:
#             # Update all columns
#             col_names = ['name1', 'name2', 'nseq', 'nad1', 'nad2', 'isot', 'd1', 'd2', 'areax', 'betax', 'sigx']
#
#         if len(data_conne) != len(self.connections):
#             # At least one connection has been added or removed, so regenerate new list of Eleme objects
#             print('The length of the numpy array data_conne does not match the number of connections in the mesh'
#                   'object.')
#             exit()
#
#         else:
#             # No connections were added or removed.  Only the attributes listed in col_names in entires from idx
#             # list have been changed
#             for i_con in idx:
#                 for col_name in col_names:
#                     # Update only columns provided in col_names
#                     setattr(self.connections[i_con], col_name, data_conne[col_name][i_con])
#
#         return None
#
#     def to_file(self, f):
#         # Print CONNE list to file 'f' (either filename or handle)
#         hdr = 'CONNE----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'
#         data_conne = self.as_numpy_array()
#         fmt = ['%5s', '%5s', '%5s', '%5s', '%5s', '%5u', '%10.4E', '%10.4E', '%10.4E', '%10.4f', '%10.4E']
#         np.savetxt(f, data_conne, header=hdr, delimiter='', fmt=fmt, comments='')
#
#         return None
#
# def gen_connections(fname):
#     """ read the fname and parse the connection data
#
#     For a description of what the parameters mean, see the TOUGH2 manual,
#     page 173.
#
#     """
#     with open(fname, 'r') as f:
#         idx =  None
#         for line in f:
#             if "CONNE" in line:
#                 idx = 0
#                 continue
#             if idx is not None:
#                 if line.strip() == "":
#                     break
#                 name = line[0:5]
#                 name2 = line[5:10]
#                 nseq = line[10:15]
#                 nad1 = line[15:20]
#                 nad2 = line[20:25]
#                 isot = int(line[25:30])
#                 d1 = float_or_none(line[30:40])
#                 d2 = float_or_none(line[40:50])
#                 areax = float_or_none(line[50:60])
#                 betax = float_or_none(line[60:70])
#                 sigx = float_or_none(line[70:80])
#                 yield Conne(name, name2, nseq, nad1, nad2, isot, d1, d2, areax, betax, sigx)
#                 idx+=1
#
# if __name__ == '__main__':
#
#     #pdir = os.path.join(".", "test_data")
#     #mesh_filename = 'SMA_ZNO_2Dhr_gv1_pv1_gas_original'
#     #pck_file = 'temp_mesh.pck'
#     pdir = os.path.join(".", "test_data_stomp")
#     mesh_filename = 'MESH_in'
#     pck_file = os.path.join(pdir, 'temp_mesh.pck')
#
#     #mesh_file = os.path.join(pdir, "SMA3Dn_R09_1_gas.mesh", "SMA3Dn_R09_1_gas.mesh")
#
#     # test_incon()
#
#     exit()
#
#     try:
#         mesh = Mesh.from_pickle(pck_file)
#         # raise Exception
#         print("Loading the mesh from a pickle ")
#     except Exception as e:
#         mesh_file = os.path.join(pdir, mesh_filename)
#         mesh = Mesh(mesh_file)
#         mesh.to_pickle(pck_file)
#         print("Pickled the mesh ")
