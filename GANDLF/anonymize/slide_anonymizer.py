#!/usr/bin/python
#
#  anonymize-slide.py - Delete the label from a whole-slide image.
#
#  Copyright (c) 2007-2013 Carnegie Mellon University
#  Copyright (c) 2011      Google, Inc.
#  Copyright (c) 2014      Benjamin Gilbert
#  All rights reserved.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of version 2 of the GNU General Public License as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor,
#  Boston, MA 02110-1301 USA.
#
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 2, June 1991

#  Copyright (C) 1989, 1991 Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

#                             Preamble

#   The licenses for most software are designed to take away your
# freedom to share and change it.  By contrast, the GNU General Public
# License is intended to guarantee your freedom to share and change free
# software--to make sure the software is free for all its users.  This
# General Public License applies to most of the Free Software
# Foundation's software and to any other program whose authors commit to
# using it.  (Some other Free Software Foundation software is covered by
# the GNU Lesser General Public License instead.)  You can apply it to
# your programs, too.

#   When we speak of free software, we are referring to freedom, not
# price.  Our General Public Licenses are designed to make sure that you
# have the freedom to distribute copies of free software (and charge for
# this service if you wish), that you receive source code or can get it
# if you want it, that you can change the software or use pieces of it
# in new free programs; and that you know you can do these things.

#   To protect your rights, we need to make restrictions that forbid
# anyone to deny you these rights or to ask you to surrender the rights.
# These restrictions translate to certain responsibilities for you if you
# distribute copies of the software, or if you modify it.

#   For example, if you distribute copies of such a program, whether
# gratis or for a fee, you must give the recipients all the rights that
# you have.  You must make sure that they, too, receive or can get the
# source code.  And you must show them these terms so they know their
# rights.

#   We protect your rights with two steps: (1) copyright the software, and
# (2) offer you this license which gives you legal permission to copy,
# distribute and/or modify the software.

#   Also, for each author's protection and ours, we want to make certain
# that everyone understands that there is no warranty for this free
# software.  If the software is modified by someone else and passed on, we
# want its recipients to know that what they have is not the original, so
# that any problems introduced by others will not reflect on the original
# authors' reputations.

#   Finally, any free program is threatened constantly by software
# patents.  We wish to avoid the danger that redistributors of a free
# program will individually obtain patent licenses, in effect making the
# program proprietary.  To prevent this, we have made it clear that any
# patent must be licensed for everyone's free use or not licensed at all.

#   The precise terms and conditions for copying, distribution and
# modification follow.

#                     GNU GENERAL PUBLIC LICENSE
#    TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

#   0. This License applies to any program or other work which contains
# a notice placed by the copyright holder saying it may be distributed
# under the terms of this General Public License.  The "Program", below,
# refers to any such program or work, and a "work based on the Program"
# means either the Program or any derivative work under copyright law:
# that is to say, a work containing the Program or a portion of it,
# either verbatim or with modifications and/or translated into another
# language.  (Hereinafter, translation is included without limitation in
# the term "modification".)  Each licensee is addressed as "you".

# Activities other than copying, distribution and modification are not
# covered by this License; they are outside its scope.  The act of
# running the Program is not restricted, and the output from the Program
# is covered only if its contents constitute a work based on the
# Program (independent of having been made by running the Program).
# Whether that is true depends on what the Program does.

#   1. You may copy and distribute verbatim copies of the Program's
# source code as you receive it, in any medium, provided that you
# conspicuously and appropriately publish on each copy an appropriate
# copyright notice and disclaimer of warranty; keep intact all the
# notices that refer to this License and to the absence of any warranty;
# and give any other recipients of the Program a copy of this License
# along with the Program.

# You may charge a fee for the physical act of transferring a copy, and
# you may at your option offer warranty protection in exchange for a fee.

#   2. You may modify your copy or copies of the Program or any portion
# of it, thus forming a work based on the Program, and copy and
# distribute such modifications or work under the terms of Section 1
# above, provided that you also meet all of these conditions:

#     a) You must cause the modified files to carry prominent notices
#     stating that you changed the files and the date of any change.

#     b) You must cause any work that you distribute or publish, that in
#     whole or in part contains or is derived from the Program or any
#     part thereof, to be licensed as a whole at no charge to all third
#     parties under the terms of this License.

#     c) If the modified program normally reads commands interactively
#     when run, you must cause it, when started running for such
#     interactive use in the most ordinary way, to print or display an
#     announcement including an appropriate copyright notice and a
#     notice that there is no warranty (or else, saying that you provide
#     a warranty) and that users may redistribute the program under
#     these conditions, and telling the user how to view a copy of this
#     License.  (Exception: if the Program itself is interactive but
#     does not normally print such an announcement, your work based on
#     the Program is not required to print an announcement.)

# These requirements apply to the modified work as a whole.  If
# identifiable sections of that work are not derived from the Program,
# and can be reasonably considered independent and separate works in
# themselves, then this License, and its terms, do not apply to those
# sections when you distribute them as separate works.  But when you
# distribute the same sections as part of a whole which is a work based
# on the Program, the distribution of the whole must be on the terms of
# this License, whose permissions for other licensees extend to the
# entire whole, and thus to each and every part regardless of who wrote it.

# Thus, it is not the intent of this section to claim rights or contest
# your rights to work written entirely by you; rather, the intent is to
# exercise the right to control the distribution of derivative or
# collective works based on the Program.

# In addition, mere aggregation of another work not based on the Program
# with the Program (or with a work based on the Program) on a volume of
# a storage or distribution medium does not bring the other work under
# the scope of this License.

#   3. You may copy and distribute the Program (or a work based on it,
# under Section 2) in object code or executable form under the terms of
# Sections 1 and 2 above provided that you also do one of the following:

#     a) Accompany it with the complete corresponding machine-readable
#     source code, which must be distributed under the terms of Sections
#     1 and 2 above on a medium customarily used for software interchange; or,

#     b) Accompany it with a written offer, valid for at least three
#     years, to give any third party, for a charge no more than your
#     cost of physically performing source distribution, a complete
#     machine-readable copy of the corresponding source code, to be
#     distributed under the terms of Sections 1 and 2 above on a medium
#     customarily used for software interchange; or,

#     c) Accompany it with the information you received as to the offer
#     to distribute corresponding source code.  (This alternative is
#     allowed only for noncommercial distribution and only if you
#     received the program in object code or executable form with such
#     an offer, in accord with Subsection b above.)

# The source code for a work means the preferred form of the work for
# making modifications to it.  For an executable work, complete source
# code means all the source code for all modules it contains, plus any
# associated interface definition files, plus the scripts used to
# control compilation and installation of the executable.  However, as a
# special exception, the source code distributed need not include
# anything that is normally distributed (in either source or binary
# form) with the major components (compiler, kernel, and so on) of the
# operating system on which the executable runs, unless that component
# itself accompanies the executable.

# If distribution of executable or object code is made by offering
# access to copy from a designated place, then offering equivalent
# access to copy the source code from the same place counts as
# distribution of the source code, even though third parties are not
# compelled to copy the source along with the object code.

#   4. You may not copy, modify, sublicense, or distribute the Program
# except as expressly provided under this License.  Any attempt
# otherwise to copy, modify, sublicense or distribute the Program is
# void, and will automatically terminate your rights under this License.
# However, parties who have received copies, or rights, from you under
# this License will not have their licenses terminated so long as such
# parties remain in full compliance.

#   5. You are not required to accept this License, since you have not
# signed it.  However, nothing else grants you permission to modify or
# distribute the Program or its derivative works.  These actions are
# prohibited by law if you do not accept this License.  Therefore, by
# modifying or distributing the Program (or any work based on the
# Program), you indicate your acceptance of this License to do so, and
# all its terms and conditions for copying, distributing or modifying
# the Program or works based on it.

#   6. Each time you redistribute the Program (or any work based on the
# Program), the recipient automatically receives a license from the
# original licensor to copy, distribute or modify the Program subject to
# these terms and conditions.  You may not impose any further
# restrictions on the recipients' exercise of the rights granted herein.
# You are not responsible for enforcing compliance by third parties to
# this License.

#   7. If, as a consequence of a court judgment or allegation of patent
# infringement or for any other reason (not limited to patent issues),
# conditions are imposed on you (whether by court order, agreement or
# otherwise) that contradict the conditions of this License, they do not
# excuse you from the conditions of this License.  If you cannot
# distribute so as to satisfy simultaneously your obligations under this
# License and any other pertinent obligations, then as a consequence you
# may not distribute the Program at all.  For example, if a patent
# license would not permit royalty-free redistribution of the Program by
# all those who receive copies directly or indirectly through you, then
# the only way you could satisfy both it and this License would be to
# refrain entirely from distribution of the Program.

# If any portion of this section is held invalid or unenforceable under
# any particular circumstance, the balance of the section is intended to
# apply and the section as a whole is intended to apply in other
# circumstances.

# It is not the purpose of this section to induce you to infringe any
# patents or other property right claims or to contest validity of any
# such claims; this section has the sole purpose of protecting the
# integrity of the free software distribution system, which is
# implemented by public license practices.  Many people have made
# generous contributions to the wide range of software distributed
# through that system in reliance on consistent application of that
# system; it is up to the author/donor to decide if he or she is willing
# to distribute software through any other system and a licensee cannot
# impose that choice.

# This section is intended to make thoroughly clear what is believed to
# be a consequence of the rest of this License.

#   8. If the distribution and/or use of the Program is restricted in
# certain countries either by patents or by copyrighted interfaces, the
# original copyright holder who places the Program under this License
# may add an explicit geographical distribution limitation excluding
# those countries, so that distribution is permitted only in or among
# countries not thus excluded.  In such case, this License incorporates
# the limitation as if written in the body of this License.

#   9. The Free Software Foundation may publish revised and/or new versions
# of the General Public License from time to time.  Such new versions will
# be similar in spirit to the present version, but may differ in detail to
# address new problems or concerns.

# Each version is given a distinguishing version number.  If the Program
# specifies a version number of this License which applies to it and "any
# later version", you have the option of following the terms and conditions
# either of that version or of any later version published by the Free
# Software Foundation.  If the Program does not specify a version number of
# this License, you may choose any version ever published by the Free Software
# Foundation.

#   10. If you wish to incorporate parts of the Program into other free
# programs whose distribution conditions are different, write to the author
# to ask for permission.  For software which is copyrighted by the Free
# Software Foundation, write to the Free Software Foundation; we sometimes
# make exceptions for this.  Our decision will be guided by the two goals
# of preserving the free status of all derivatives of our free software and
# of promoting the sharing and reuse of software generally.

#                             NO WARRANTY

#   11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
# FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
# OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
# PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
# OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
# TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
# PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
# REPAIR OR CORRECTION.

#   12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
# WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
# REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
# INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
# OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
# TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
# YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
# PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGES.

#                      END OF TERMS AND CONDITIONS

#             How to Apply These Terms to Your New Programs

#   If you develop a new program, and you want it to be of the greatest
# possible use to the public, the best way to achieve this is to make it
# free software which everyone can redistribute and change under these terms.

#   To do so, attach the following notices to the program.  It is safest
# to attach them to the start of each source file to most effectively
# convey the exclusion of warranty; and each file should have at least
# the "copyright" line and a pointer to where the full notice is found.

#     <one line to give the program's name and a brief idea of what it does.>
#     Copyright (C) <year>  <name of author>

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Also add information on how to contact you by electronic and paper mail.

# If the program is interactive, make it output a short notice like this
# when it starts in an interactive mode:

#     Gnomovision version 69, Copyright (C) year name of author
#     Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.

# The hypothetical commands `show w' and `show c' should show the appropriate
# parts of the General Public License.  Of course, the commands you use may
# be called something other than `show w' and `show c'; they could even be
# mouse-clicks or menu items--whatever suits your program.

# You should also get your employer (if you work as a programmer) or your
# school, if any, to sign a "copyright disclaimer" for the program, if
# necessary.  Here is a sample; alter the names:

#   Yoyodyne, Inc., hereby disclaims all copyright interest in the program
#   `Gnomovision' (which makes passes at compilers) written by James Hacker.

#   <signature of Ty Coon>, 1 April 1989
#   Ty Coon, President of Vice

# This General Public License does not permit incorporating your program into
# proprietary programs.  If your program is a subroutine library, you may
# consider it more useful to permit linking proprietary applications with the
# library.  If this is what you want to do, use the GNU Lesser General
# Public License instead of this License.
#
# This is an adaptation of the code from :
# https://github.com/bgilbert/anonymize-slide


from __future__ import division

from configparser import RawConfigParser
from glob import glob
from optparse import OptionParser
import os
import struct
import sys
import io
import shutil

PROG_DESCRIPTION = '''
Delete the slide label from an MRXS, NDPI, or SVS whole-slide image.
'''.strip()
PROG_VERSION = '1.1.1'
DEBUG = False

# TIFF types
ASCII = 2
SHORT = 3
LONG = 4
FLOAT = 11
DOUBLE = 12
LONG8 = 16

# TIFF tags
IMAGE_DESCRIPTION = 270
STRIP_OFFSETS = 273
STRIP_BYTE_COUNTS = 279
NDPI_MAGIC = 65420
NDPI_SOURCELENS = 65421

# Format headers
LZW_CLEARCODE = b'\x80'
JPEG_SOI = b'\xff\xd8'
UTF8_BOM = b'\xef\xbb\xbf'

# MRXS
MRXS_HIERARCHICAL = 'HIERARCHICAL'
MRXS_NONHIER_ROOT_OFFSET = 41


class UnrecognizedFile(Exception):
    pass


class TiffFile(io.FileIO):
    def __init__(self, path, mode='r+b', *args, **kwargs):
        super(TiffFile, self).__init__(path, mode)

        # Check header, decide endianness
        endian = self.read(2).decode('utf-8')
        if endian == 'II':
            self._fmt_prefix = '<'
        elif endian == 'MM':
            self._fmt_prefix = '>'
        else:
            raise UnrecognizedFile

        # Check TIFF version
        self._bigtiff = False
        self._ndpi = False
        version = self.read_fmt('H')
        if version == 42:
            pass
        elif version == 43:
            self._bigtiff = True
            magic2, reserved = self.read_fmt('HH')
            if magic2 != 8 or reserved != 0:
                raise UnrecognizedFile
        else:
            raise UnrecognizedFile

        # Read directories
        self.directories = []
        while True:
            in_pointer_offset = self.tell()
            directory_offset = self.read_fmt('D')
            if directory_offset == 0:
                break
            self.seek(directory_offset)
            directory = TiffDirectory(self, len(self.directories),
                    in_pointer_offset)
            if not self.directories and not self._bigtiff:
                # Check for NDPI.  Because we don't know we have an NDPI file
                # until after reading the first directory, we will choke if
                # the first directory is beyond 4 GB.
                if NDPI_MAGIC in directory.entries:
                    if DEBUG:
                        print('Enabling NDPI mode.')
                    self._ndpi = True
            self.directories.append(directory)
        if not self.directories:
            raise IOError('No directories')

    def _convert_format(self, fmt):
        # Format strings can have special characters:
        # y: 16-bit   signed on little TIFF, 64-bit   signed on BigTIFF
        # Y: 16-bit unsigned on little TIFF, 64-bit unsigned on BigTIFF
        # z: 32-bit   signed on little TIFF, 64-bit   signed on BigTIFF
        # Z: 32-bit unsigned on little TIFF, 64-bit unsigned on BigTIFF
        # D: 32-bit unsigned on little TIFF, 64-bit unsigned on BigTIFF/NDPI
        if self._bigtiff:
            fmt = fmt.translate(str.maketrans('yYzZD', 'qQqQQ'))
        elif self._ndpi:
            fmt = fmt.translate(str.maketrans('yYzZD', 'hHiIQ'))
        else:
            fmt = fmt.translate(str.maketrans('yYzZD', 'hHiII'))
        return self._fmt_prefix + fmt

    def fmt_size(self, fmt):
        return struct.calcsize(self._convert_format(fmt))

    def near_pointer(self, base, offset):
        # If NDPI, return the value whose low-order 32-bits are equal to
        # @offset and which is within 4 GB of @base and below it.
        # Otherwise, return offset.
        if self._ndpi and offset < base:
            seg_size = 1 << 32
            offset += ((base - offset) // seg_size) * seg_size
        return offset

    def read_fmt(self, fmt, force_list=False):
        fmt = self._convert_format(fmt)
        vals = struct.unpack(fmt, self.read(struct.calcsize(fmt)))
        if len(vals) == 1 and not force_list:
            return vals[0]
        else:
            return vals

    def write_fmt(self, fmt, *args):
        fmt = self._convert_format(fmt)
        self.write(struct.pack(fmt, *args))


class TiffDirectory(object):
    def __init__(self, fh, number, in_pointer_offset):
        self.entries = {}
        count = fh.read_fmt('Y')
        for _ in range(count):
            entry = TiffEntry(fh)
            self.entries[entry.tag] = entry
        self._in_pointer_offset = in_pointer_offset
        self._out_pointer_offset = fh.tell()
        self._fh = fh
        self._number = number

    def delete(self, expected_prefix=None):
        # Get strip offsets/lengths
        try:
            offsets = self.entries[STRIP_OFFSETS].value()
            lengths = self.entries[STRIP_BYTE_COUNTS].value()
        except KeyError:
            raise IOError('Directory is not stripped')

        # Wipe strips
        for offset, length in zip(offsets, lengths):
            offset = self._fh.near_pointer(self._out_pointer_offset, offset)
            if DEBUG:
                print('Zeroing', offset, 'for', length)
            self._fh.seek(offset)

            # This was taken out because the buffers wouldn't line up correctly.
            if expected_prefix:
                buf = self._fh.read(len(expected_prefix))
                if buf != expected_prefix:
                    raise IOError('Unexpected data in image strip')
                self._fh.seek(offset)

            self._fh.write(b'\x00' * length)

        # Remove directory
        if DEBUG:
            print('Deleting directory', self._number)
        self._fh.seek(self._out_pointer_offset)
        out_pointer = self._fh.read_fmt('D')
        self._fh.seek(self._in_pointer_offset)
        self._fh.write_fmt('D', out_pointer)


class TiffEntry(object):
    def __init__(self, fh):
        self.start = fh.tell()
        self.tag, self.type, self.count, self.value_offset = \
                fh.read_fmt('HHZZ')
        self._fh = fh

    def value(self):
        if self.type == ASCII:
            item_fmt = 'c'
        elif self.type == SHORT:
            item_fmt = 'H'
        elif self.type == LONG:
            item_fmt = 'I'
        elif self.type == LONG8:
            item_fmt = 'Q'
        elif self.type == FLOAT:
            item_fmt = 'f'
        elif self.type == DOUBLE:
            item_fmt = 'd'
        else:
            raise ValueError('Unsupported type')

        fmt = '%d%s' % (self.count, item_fmt)
        len = self._fh.fmt_size(fmt)
        if len <= self._fh.fmt_size('Z'):
            # Inline value
            self._fh.seek(self.start + self._fh.fmt_size('HHZ'))
        else:
            # Out-of-line value
            self._fh.seek(self._fh.near_pointer(self.start, self.value_offset))
        items = self._fh.read_fmt(fmt, force_list=True)
        if self.type == ASCII:
            if items[-1] != b'\x00':
                raise ValueError('String not null-terminated')
            return ''.join([item.decode('utf-8') for item in items[:-1]])
        else:
            return items


class MrxsFile(object):
    def __init__(self, filename):
        # Split filename
        dirname, ext = os.path.splitext(filename)
        if ext != '.mrxs':
            raise UnrecognizedFile

        # Parse slidedat
        self._slidedatfile = os.path.join(dirname, 'Slidedat.ini')
        self._dat = RawConfigParser()
        self._dat.optionxform = str
        try:
            with open(self._slidedatfile, 'rb') as fh:
                self._have_bom = (fh.read(len(UTF8_BOM)) == UTF8_BOM)
                if not self._have_bom:
                    fh.seek(0)
                self._dat.readfp(fh)
        except IOError:
            raise UnrecognizedFile

        # Get file paths
        self._indexfile = os.path.join(dirname,
                self._dat.get(MRXS_HIERARCHICAL, 'INDEXFILE'))
        self._datafiles = [os.path.join(dirname,
                self._dat.get('DATAFILE', 'FILE_%d' % i))
                for i in range(self._dat.getint('DATAFILE', 'FILE_COUNT'))]

        # Build levels
        self._make_levels()

    def _make_levels(self):
        self._levels = {}
        self._level_list = []
        layer_count = self._dat.getint(MRXS_HIERARCHICAL, 'NONHIER_COUNT')
        for layer_id in range(layer_count):
            level_count = self._dat.getint(MRXS_HIERARCHICAL,
                    'NONHIER_%d_COUNT' % layer_id)
            for level_id in range(level_count):
                level = MrxsNonHierLevel(self._dat, layer_id, level_id,
                        len(self._level_list))
                self._levels[(level.layer_name, level.name)] = level
                self._level_list.append(level)

    @classmethod
    def _read_int32(cls, f):
        buf = f.read(4)
        if len(buf) != 4:
            raise IOError('Short read')
        return struct.unpack('<i', buf)[0]

    @classmethod
    def _assert_int32(cls, f, value):
        v = cls._read_int32(f)
        if v != value:
            raise ValueError('%d != %d' % (v, value))

    def _get_data_location(self, record):
        with open(self._indexfile, 'rb') as fh:
            fh.seek(MRXS_NONHIER_ROOT_OFFSET)
            # seek to record
            table_base = self._read_int32(fh)
            fh.seek(table_base + record * 4)
            # seek to list head
            list_head = self._read_int32(fh)
            fh.seek(list_head)
            # seek to data page
            self._assert_int32(fh, 0)
            page = self._read_int32(fh)
            fh.seek(page)
            # check pagesize
            self._assert_int32(fh, 1)
            # read rest of prologue
            self._read_int32(fh)
            self._assert_int32(fh, 0)
            self._assert_int32(fh, 0)
            # read values
            position = self._read_int32(fh)
            size = self._read_int32(fh)
            fileno = self._read_int32(fh)
            return (self._datafiles[fileno], position, size)

    def _zero_record(self, record):
        path, offset, length = self._get_data_location(record)
        with open(path, 'r+b') as fh:
            fh.seek(0, 2)
            do_truncate = (fh.tell() == offset + length)
            if DEBUG:
                if do_truncate:
                    print('Truncating', path, 'to', offset)
                else:
                    print('Zeroing', path, 'at', offset, 'for', length)
            fh.seek(offset)
            buf = fh.read(len(JPEG_SOI))
            if buf != JPEG_SOI:
                raise IOError('Unexpected data in nonhier image')
            if do_truncate:
                fh.truncate(offset)
            else:
                fh.seek(offset)
                fh.write('\0' * length)

    def _delete_index_record(self, record):
        if DEBUG:
            print('Deleting record', record)
        with open(self._indexfile, 'r+b') as fh:
            entries_to_move = len(self._level_list) - record - 1
            if entries_to_move == 0:
                return
            # get base of table
            fh.seek(MRXS_NONHIER_ROOT_OFFSET)
            table_base = self._read_int32(fh)
            # read tail of table
            fh.seek(table_base + (record + 1) * 4)
            buf = fh.read(entries_to_move * 4)
            if len(buf) != entries_to_move * 4:
                raise IOError('Short read')
            # overwrite the target record
            fh.seek(table_base + record * 4)
            fh.write(buf)

    def _hier_keys_for_level(self, level):
        ret = []
        for k, _ in self._dat.items(MRXS_HIERARCHICAL):
            if k == level.key_prefix or k.startswith(level.key_prefix + '_'):
                ret.append(k)
        return ret

    def _rename_section(self, old, new):
        if self._dat.has_section(old):
            if DEBUG:
                print('[%s] -> [%s]' % (old, new))
            self._dat.add_section(new)
            for k, v in self._dat.items(old):
                self._dat.set(new, k, v)
            self._dat.remove_section(old)
        elif DEBUG:
            print('[%s] does not exist' % old)

    def _delete_section(self, section):
        if DEBUG:
            print('Deleting [%s]' % section)
        self._dat.remove_section(section)

    def _set_key(self, section, key, value):
        if DEBUG:
            prev = self._dat.get(section, key)
            print('[%s] %s: %s -> %s' % (section, key, prev, value))
        self._dat.set(section, key, value)

    def _rename_key(self, section, old, new):
        if DEBUG:
            print('[%s] %s -> %s' % (section, old, new))
        v = self._dat.get(section, old)
        self._dat.remove_option(section, old)
        self._dat.set(section, new, v)

    def _delete_key(self, section, key):
        if DEBUG:
            print('Deleting [%s] %s' % (section, key))
        self._dat.remove_option(section, key)

    def _write(self):
        buf = io.StringIO()
        self._dat.write(buf)
        with open(self._slidedatfile, 'wb') as fh:
            if self._have_bom:
                fh.write(bytearray(UTF8_BOM))
            fh.write(bytearray(buf.getvalue().replace('\n', '\r\n')))

    def delete_level(self, layer_name, level_name):
        level = self._levels[(layer_name, level_name)]
        record = level.record

        # Zero image data
        self._zero_record(record)

        # Delete pointer from nonhier table in index
        self._delete_index_record(record)

        # Remove slidedat keys
        for k in self._hier_keys_for_level(level):
            self._delete_key(MRXS_HIERARCHICAL, k)

        # Remove slidedat section
        self._delete_section(level.section)

        # Rename section and keys for subsequent levels in the layer
        prev_level = level
        for cur_level in self._level_list[record + 1:]:
            if cur_level.layer_id != prev_level.layer_id:
                break
            for k in self._hier_keys_for_level(cur_level):
                new_k = k.replace(cur_level.key_prefix,
                        prev_level.key_prefix, 1)
                self._rename_key(MRXS_HIERARCHICAL, k, new_k)
            self._set_key(MRXS_HIERARCHICAL, prev_level.section_key,
                    prev_level.section)
            self._rename_section(cur_level.section, prev_level.section)
            prev_level = cur_level

        # Update level count within layer
        count_k = 'NONHIER_%d_COUNT' % level.layer_id
        count_v = self._dat.getint(MRXS_HIERARCHICAL, count_k)
        self._set_key(MRXS_HIERARCHICAL, count_k, count_v - 1)

        # Write slidedat
        self._write()

        # Refresh metadata
        self._make_levels()


class MrxsNonHierLevel(object):
    def __init__(self, dat, layer_id, level_id, record):
        self.layer_id = layer_id
        self.id = level_id
        self.record = record
        self.layer_name = dat.get(MRXS_HIERARCHICAL,
                'NONHIER_%d_NAME' % layer_id)
        self.key_prefix = 'NONHIER_%d_VAL_%d' % (layer_id, level_id)
        self.name = dat.get(MRXS_HIERARCHICAL, self.key_prefix)
        self.section_key = self.key_prefix + '_SECTION'
        self.section = dat.get(MRXS_HIERARCHICAL, self.section_key)


def accept(filename, format):
    if DEBUG:
        print(filename + ':', format)


def do_aperio_svs(filename):
    with TiffFile(filename) as fh:
        # Check for SVS file
        try:
            desc0 = fh.directories[0].entries[IMAGE_DESCRIPTION].value()
            if not desc0.startswith('Aperio'):
                raise UnrecognizedFile
        except KeyError:
            raise UnrecognizedFile
        accept(filename, 'SVS')

        # Find and delete label
        for directory in fh.directories:
            lines = directory.entries[IMAGE_DESCRIPTION].value().splitlines()
            if len(lines) >= 2 and lines[1].startswith('label '):
                directory.delete(expected_prefix=LZW_CLEARCODE)
                break
        else:
            raise IOError("No label in SVS file")


def do_hamamatsu_ndpi(filename):
    with TiffFile(filename) as fh:
        # Check for NDPI file
        if NDPI_MAGIC not in fh.directories[0].entries:
            raise UnrecognizedFile
        accept(filename, 'NDPI')

        # Find and delete macro image
        for directory in fh.directories:
            if directory.entries[NDPI_SOURCELENS].value()[0] == -1:
                directory.delete(expected_prefix=JPEG_SOI)
                break
        else:
            raise IOError("No label in NDPI file")


def do_3dhistech_mrxs(filename):
    mrxs = MrxsFile(filename)
    try:
        mrxs.delete_level('Scan data layer', 'ScanDataLayer_SlideBarcode')
    except KeyError:
        raise IOError('No label in MRXS file')


format_handlers = [
    do_aperio_svs,
    do_hamamatsu_ndpi,
    do_3dhistech_mrxs,
]


def anonymize_slide(input_path, output_path):

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    shutil.copyfile(input_path, output_path)
    filename = output_path

    exit_code = 0
    try:
        for handler in format_handlers:
            try:
                handler(filename)
                break
            except UnrecognizedFile:
                pass
        else:
            raise IOError('Unrecognized file type')
    except Exception as e:
        if DEBUG:
            raise
        print(sys.stderr, '%s: %s' % (filename, str(e)))
        exit_code = 1
    sys.exit(exit_code)
