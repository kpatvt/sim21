#!/usr/bin/python

"""
Units module - contains classes and methods for doing unit conversions.
"""
import re
import math
from importlib.resources import open_text, contents

EMPTY_VAL = -12321
InternalUnitItemOffset = 10000

ConvertToBaseOps = [lambda value, scale, offset: scale * value + offset,
                    lambda value, scale, offset: scale / value + offset,
                    lambda value, scale, offset: scale * value * value + offset,
                    lambda value, scale, offset: scale / value / value + offset,
                    lambda value, scale, offset: offset,
                    lambda value, scale, offset: scale / (value + offset)]

ConvertFromBaseOps = [lambda value, scale, offset: (value - offset) / scale,
                      lambda value, scale, offset: scale / (value - offset),
                      lambda value, scale, offset: math.sqrt((value - offset) / scale),
                      lambda value, scale, offset: math.sqrt(scale / (value - offset)),
                      lambda value, scale, offset: offset,
                      lambda value, scale, offset: (scale / value) - offset]

OperationRenderings = ['%f * (%s) + %f', '%f / (%s) + %f', '%f * (%s)^2 + %f', '%f / (%s)^2 + %f', '%f * (%s) + %f',
                       '%f / (%s + %f)']
OperationsScaleOnly = ['%f * (%s)', '%f / (%s)', '%f * (%s)^2', '%f / (%s)^2', '%f', '%f * (%s)']

# this global variable can be externally set to indicate where the unit
# data files are kept - particularly useful for frozen apps.
globalBasePath = None


class UnitItem:
    """Basic unit in unit system"""

    def __init__(self, unitSystem):
        """initialize attributes"""
        self.unitSystem = unitSystem
        self.id = 0
        self.typeID = 0
        self.name = 'Unknown'
        self.scale = 1.0
        self.offset = 0.0
        self.operation = 1
        self.notes = 'Unknown'

    def CleanUp(self):
        self.unitSystem = None

    def ReadFile(self, f):
        """
        read a line from a tab separated file to initialize values
        return the type id if successful or None if not
        """
        line = f.readline()
        if not line:
            return None

        fields = re.split(r'\t', line.strip())

        if fields[0] == 'ID':  # check for header line
            line = f.readline()
            if not line:
                return None
            fields = re.split(r'\t', line.strip())

        self.id = int(fields[0])
        self.typeID = int(fields[1])
        self.name = fields[2]
        self.scale = float(fields[3])
        self.offset = float(fields[4])
        self.operation = int(fields[5])
        try:
            self.notes = fields[6]
        except IndexError:
            self.notes = ''
        return self.id

    def WriteFile(self, f):
        """
        write a line to a tab separated file
        """
        f.write('%d\t%d\t%s\t%e\t%e\t%d\t%s\n' % (self.id, self.typeID, self.name,
                                                  self.scale, self.offset,
                                                  self.operation, self.notes))

    def ConvertFromBase(self, value):
        if value is None:
            return None
        return ConvertFromBaseOps[self.operation - 1](value, self.scale, self.offset)

    def ConvertToBase(self, value):
        if value is None:
            return None
        return ConvertToBaseOps[self.operation - 1](value, self.scale, self.offset)

    def ConvertToSim42(self, value):
        baseValue = self.ConvertToBase(value)
        sim42Unit = self.unitSystem.GetSim42Unit(self.typeID)
        return sim42Unit.ConvertFromBase(baseValue)

    def ConvertFromSim42(self, value):
        if value == EMPTY_VAL:
            return EMPTY_VAL
        sim42Unit = self.unitSystem.GetSim42Unit(self.typeID)
        baseValue = sim42Unit.ConvertToBase(value)
        return self.ConvertFromBase(baseValue)

    def ConvertToSet(self, setName, value):
        baseValue = self.ConvertToBase(value)
        unit_type = self.unitSystem.GetUnit(self.unitSystem.GetUnitSet(setName), self.typeID)
        return unit_type.ConvertFromBase(baseValue)

    def ConvertFromSet(self, setName, value):
        unit_type = self.unitSystem.GetUnit(self.unitSystem.GetUnitSet(setName), self.typeID)
        baseValue = unit_type.ConvertToBase(value)
        return self.ConvertFromBase(baseValue)

    def RenderOperation(self):
        """return a string representing the operation"""
        s = 'Base = '
        if self.offset == 0.0:
            s += OperationsScaleOnly[self.operation - 1] % (self.scale, self.name)
        elif self.operation == 5:
            s += '%f' % self.offset
        else:
            s += OperationRenderings[self.operation - 1] % (self.scale, self.name, self.offset)
        return s

    def Clone(self, standard=0):
        """ return a clone of self """
        clone = UnitItem(self.unitSystem)
        clone.id = 0  # dummy until added to list
        clone.typeID = self.typeID
        clone.name = '%s-clone' % self.name
        clone.scale = self.scale
        clone.offset = self.offset
        clone.operation = self.operation
        clone.notes = self.notes
        return clone


class UnitType:
    """type of unit"""

    def __init__(self):
        """initialize attributes"""
        self.id = 0
        self.name = 'Unknown'
        self.equivalentType = None

    def ReadFile(self, f):
        """
        read a line from a tab separated file to initialize values
        return the type id if successful or None if not
        """
        line = f.readline()
        if not line:
            return None

        fields = re.split(r'\t', line.strip())
        if fields[0] == 'ID':  # check for header line
            line = f.readline()
            if not line:
                return None
            fields = re.split(r'\t', line.strip())

        self.id = int(fields[0])
        self.name = fields[1]
        self.equivalentType = self.id
        return self.id

    def WriteFile(self, f):
        """
        write a line to a tab separated file
        """
        f.write('%d\t%s\n' % (self.id, self.name))


class UnitSet(dict):
    """
    Dictionary of unit ids stored by type key that represent the
    default units for the set
    """

    def __init__(self, isMaster=0):
        """
        The isMaster flag indicates whether this is one of the
        distributed master sets or a user set
        """
        dict.__init__(self)
        self.isMaster = isMaster


class UnitSystem:
    """base class for containing the units and unit sets"""

    def __init__(self, userDir=None):
        """
        read in the unit system information and set things up
        userDir is the path to the directory containing user data, if any
        """
        self.baseDataPath = None
        self.userDataPath = userDir

        # read in types
        self.types = {}
        # first base types
        # fileName = self.baseDataPath + os.sep + 'UnitType.txt'
        f = open_text('sim21.uom.data', 'UnitType.txt')
        while 1:
            unitType = UnitType()
            unit_id = unitType.ReadFile(f)
            if unit_id is None:
                break
            self.types[unit_id] = unitType

        # check if user types exist
        # REMOVED

        # read in unit items
        self.units = {}
        # fileName = self.baseDataPath + os.sep + 'UnitItem.txt'
        f = open_text('sim21.uom.data', 'UnitItem.txt')
        while 1:
            unit_type = UnitItem(self)
            unit_id = unit_type.ReadFile(f)
            if unit_id is None:
                break
            self.units[unit_id] = unit_type

        # check if user items exist

        # create cross reference for quick look up by name
        self.nameIndex = {}
        for unit_type in list(self.units.values()):
            self.nameIndex[unit_type.name] = unit_type.id

        # read standard unit sets
        self.unitSets = {}
        self.ReadSets()

        # see if there is a current default unit set
        self.defaultSet = self.unitSets['SI']
        self.sim42Set = self.unitSets.get('sim42', None)

        # fix up equivalent unit types after creation of nameIndex
        self.FixEquivalentTypes()

    def CleanUp(self):
        [u.CleanUp() for u in list(self.units.values())]
        self.unitSets = {}
        self.units = {}

    def ReadSets(self):
        """
        read in unit sets from dirName
        The isMaster flag indicates whether these are the
        distributed master sets or a user sets
        """
        files = contents('sim21.uom.data')
        sets = [file for file in files if re.search(r'\.set$', file)]
        for set_name in sets:
            setName = re.sub(r'\.set$', '', set_name)
            self.unitSets[setName] = self.ReadSet(set_name)

    def ReadSet(self, fileName):
        """
        read in unit set from file described by fileName
        return the new set which is a dictionary where the key is
        the typeID and the value is the unitID
        The isMaster flag indicates whether this is one of the
        distributed master sets or a user set
        """
        f = open_text('sim21.uom.data', fileName)
        set_name = UnitSet(1)
        while 1:
            line = f.readline()
            if not line:
                break
            fields = re.split('\t', line.strip())
            set_name[int(fields[0])] = int(fields[1])

        return set_name

    def WriteSets(self, dirName, isMaster):
        """
        write out unit sets from dirName
        The isMaster flag indicates whether these are the
        distributed master sets or a user sets
        """
        raise NotImplemented

    def WriteSet(self, fileName, set_name):
        """
        write out unit set to fileName
        """
        raise NotImplemented

    def Write(self, master=0):
        """
        write out information back to their files
        if master is true, then write to master files
        otherwise user files
        """
        raise NotImplementedError

    def AddSet(self, setName, set_name):
        """add set to set collection using setName as key"""
        self.unitSets[setName] = set_name

    def DeleteSet(self, setName):
        """delete set references by setName from the set collection"""
        del self.unitSets[setName]

    def GetSetNames(self):
        """return the available unit sets as a list of names"""
        return list(self.unitSets.keys())

    def GetBaseSetNames(self):
        """return the names of the sets that are always there"""
        return ['PureSI', 'VMG', 'Yaws', 'Hysys', 'British', 'DIPPR', 'Field', 'SI', 'sim42']

    def GetUnits(self):
        """return a list of all units"""
        return list(self.units.values())

    def GetUnitIDs(self):
        return list(self.units.keys())

    def IsEquivalentType(self, typeID1, typeID2):
        """
        Test whether unit type 1 and unit type 2 are of identical types.
        Types are identical if their corresponding unit items are equal in the Sim42.set
        e.g. Power and Work are identical unit types
        """
        id1 = self.units[self.sim42Set[typeID1]].id
        id2 = self.units[self.sim42Set[typeID2]].id
        return id1 == id2

    def FixEquivalentTypes(self):
        """
        Some unit types share the same list of unit items, duplicate these items for simplicity
        Offset their ID's by a large number to avoid conflicts
        """
        for typeId in list(self.types.keys()):
            # test each unit type
            items = self.UnitsByType(typeId)
            eqType = typeId
            if len(items) == 0:
                # has no unit items of this type, find the equivalent id
                s42UnitItemId = self.sim42Set[typeId]

                ui = self.units[s42UnitItemId]
                equivalentTypeID = ui.typeID
                # get all the unit items of the equivalent type
                items = list(filter(lambda unit_id, type_id=equivalentTypeID: unit_id.typeID == type_id, list(self.units.values())))
                # get these unit items
                for ui in items:
                    unit_type = ui.Clone()
                    unit_type.id = ui.id + InternalUnitItemOffset
                    unit_type.typeID = typeId
                    unit_type.name = ui.name
                    # add the clone to the unit item dictionary
                    self.units[unit_type.id] = unit_type

                # add an equivalent type Id to the unit type
                # Search the Sim42 unit set for another unit type having the same unit item ID
                for t in list(self.sim42Set.keys()):
                    if t != typeId:
                        if self.sim42Set[t] == s42UnitItemId:
                            eqType = t
                            break

            self.types[typeId].equivalentType = eqType
            items = self.UnitsByType(typeId)
            pass

    def UnitsByType(self, type_id):
        """return list units with type typeID"""
        if type_id > 0:
            type_id = self.types[type_id].equivalentType

        return list(filter(lambda unit_id, t=type_id: unit_id.typeID == t, list(self.units.values())))

    def UnitsByPartialName(self, name, type_id=None):
        """return list of units that match name and typeID (if it is not None)"""
        name = name.replace("(", "\\(")
        name = name.replace(")", "\\)")
        if type_id:
            if type_id > 0:
                type_id = self.types[type_id].equivalentType
            return list(filter(lambda u, n=name, t=type_id: u.typeID == t and re.match(n, u.name, re.I),
                               list(self.units.values())))
        else:
            return list(filter(lambda u, n=name: re.match(n, u.name, re.I), list(self.units.values())))

    def GetUnitSet(self, setName):
        return self.unitSets[setName]

    def SetDefaultSet(self, setName):
        """Set the default unit set"""
        given_set = self.unitSets[setName]
        self.defaultSet = given_set
        return True

    def GetDefaultSet(self):
        """Gets the default unit set name"""
        uDef = None
        for name, given_set in list(self.unitSets.items()):
            if self.defaultSet == given_set:
                uDef = name
                break

        return uDef

    def GetType(self, type_id):
        """return the type corresponding to typeID"""
        return self.types[type_id]

    def GetTypeID(self, typeName):
        """returns type id from a name"""
        typeIDs = list(filter(lambda t, n=typeName: t.name == n, list(self.types.values())))
        # to get around the simcom installation problem
        #    when installing a new version with a new unit type
        if typeIDs and len(typeIDs) > 0:
            return typeIDs[0].id
        else:
            return None

    def GetTypes(self):
        """return list of all types"""
        return list(self.types.values())

    def GetTypeIDs(self):
        """return list of type ids"""
        return list(self.types.keys())

    def GetTypeName(self, type_id):
        """returns name of type with id typeID"""
        return self.types[type_id].name

    def DeleteType(self, type_id):
        """deletes type associated with type ID and all the units that have that type"""
        units_by_type = self.UnitsByType(type_id)
        for u in units_by_type:
            self.DeleteUnit(u.id)
        del self.types[type_id]
        for given_set in list(self.unitSets.values()):
            del given_set[type_id]

    def GetUnit(self, unitSet, type_id):
        """returns the unit in unitSet set for typeID. If not available, then return sim42 unit"""
        if type_id is None:
            return None

        try:
            unitID = unitSet[type_id]
        except KeyError:
            unitID = self.sim42Set[type_id]

        try:
            return self.units[unitID]
        except KeyError:
            return None

    def GetUnitWithID(self, id_name):
        return self.units[id_name]

    def GetCurrentUnit(self, type_id):
        """returns unit in default unit set with type typeID. If unit is missing then return sim42 unit"""
        if type_id is None or self.defaultSet is None:
            return None
        try:
            unitID = self.defaultSet[type_id]
        except KeyError:
            try:
                unitID = self.sim42Set[type_id]
            except KeyError:
                return None

        return self.units[unitID]

    def GetSim42Unit(self, type_id):
        """returns the sim42 property package unit with type typeID"""
        if type_id is None or self.sim42Set is None:
            return None
        return self.units[self.sim42Set[type_id]]

    def GetUserDir(self):
        """return the current user directory"""
        return self.userDataPath

    def SetUserDir(self, path):
        """set the userDataPath to path"""
        # self.userDataPath = path
        # # reread unit sets
        # self.unitSets = {}
        # self.ReadSets(self.baseDataPath, 1)
        # self.ReadSets(self.userDataPath, 0)
        raise NotImplementedError

    def AddUserType(self, newType):
        """add a new user unit type to the self.types - create negative id"""
        id_int = min(self.types.keys()) - 1
        id_int = min(id_int, -1)  # in case there were no other user types
        newType.id = id_int
        self.types[id_int] = newType

        firstUnit = UnitItem(self)
        firstUnit.typeID = id_int
        unitID = self.AddUserUnit(firstUnit)
        for given_set in list(self.unitSets.values()):
            if not given_set.isMaster:
                given_set[id_int] = unitID
        return id_int

    def AddMasterType(self, newType):
        """add a new master unit type to the self.types - create positive id"""
        id_int = max(self.types.keys()) + 1
        id_int = max(id_int, 1)  # in case there were no other types
        newType.id = id_int
        self.types[id_int] = newType
        firstUnit = UnitItem(self)
        firstUnit.typeID = id_int
        unitID = self.AddMasterUnit(firstUnit)
        for given_set in list(self.unitSets.values()):
            if given_set.isMaster:
                given_set[id_int] = unitID
        return id_int

    def AddUserUnit(self, unit_obj):
        """add a new user unit to the self.units - create negative id"""
        given_id = min(self.units.keys()) - 1
        given_id = min(given_id, -1)  # in case there were no other user units
        unit_obj.id = given_id
        self.units[given_id] = unit_obj
        return given_id

    def AddMasterUnit(self, unit_obj):
        """add a new master unit to the self.units - create positive id"""
        given_id = max(self.units.keys()) + 1
        given_id = max(given_id, 1)  # in case there were no other units
        unit_obj.id = given_id
        self.units[given_id] = unit_obj
        return given_id

    def ReplaceUnit(self, unit_obj):
        """use the id of unit as key for unit"""
        self.units[unit_obj.id] = unit_obj

    def DeleteUnit(self, unitID):
        """delete the unit with id unitID"""
        del self.units[unitID]


if __name__ == '__main__':
    units = UnitSystem()
    for i in list(units.types.values()):
        print((i.name, i.id))

    print('Units')
    for unit in units.units.values():
        print(unit.name, unit.scale, unit.offset, unit.operation, unit.notes)

    for i in units.GetSetNames():
        print(i)

    for unit in units.UnitsByType(9):
        print((unit.name, unit.scale, unit.notes))

    for unit in units.UnitsByPartialName('k', 32):
        print((unit.name, unit.typeID, unit.scale, unit.notes))
        base = unit.ConvertToBase(1.0)
        print((base, unit.ConvertFromBase(base)))

    print(('Type ID for MoleFlow', units.GetTypeID('MoleFlow')))
    typeID = units.GetTypeID('Temperature')
    fieldSet = units.GetUnitSet('Field')
    print(('Field unit for Temperature', units.GetUnit(fieldSet, typeID).name))
    pass
