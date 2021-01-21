"""
Sim21 interface for the simulator

Classes:
ThermoInterface -- Main class of the interfase

"""
from sim21.data import chemsep, twu
from sim21.provider.cubic import PengRobinson, SoaveRedlichKwong
from sim21.provider.error import FlashConvergenceError
from sim21.solver.Messages import MessageHandler
from sim21.solver.Variables import *
from sim21.thermo import Oils
from sim21.thermo.ThermoConstants import *
from sim21.solver.Error import SimError
from sim21.thermo.ThermoAdmin import FlashResults, EnvelopeResults, ThermoCase
from sim21.thermo.Hypo import *
import copy

# Some constants to handle some special property packages
SOLID_MODELS = ['SimpleSolid']
EMPTY_MODEL = 'Unknown'


def extract_property_from_phase(prop_name, ph):
    if prop_name == T_VAR:
        new_value = float(ph.temp)
    elif prop_name == P_VAR:
        new_value = float(ph.press) * 1e-3
    elif prop_name == H_VAR:
        new_value = float(ph.enthalpy_mole) * 1e-3
    elif prop_name == S_VAR:
        new_value = float(ph.entropy_mole) * 1e-3
    elif prop_name == VPFRAC_VAR:
        new_value = float(ph.vap_frac_mole)
    elif prop_name == molarV_VAR:
        new_value = float(ph.vol_mole)
    elif prop_name == MASSDEN_VAR:
        new_value = float(ph.dens_mass)
    elif prop_name == ZFACTOR_VAR:
        new_value = float(ph.z_factor)
    elif prop_name == MOLEWT_VAR:
        new_value = float(ph.mw)
    elif prop_name == STDLIQVOL_VAR:
        new_value = float(ph.std_liq_vol_mole)
    elif prop_name == CP_VAR:
        new_value = float(ph.cp_mole) * 1e-3
    elif prop_name == CV_VAR:
        new_value = float(ph.cv_mole) * 1e-3
    elif prop_name == VISCOSITY_VAR:
        new_value = float(ph.visc)
    elif prop_name == IDEALGASENTHALPY_VAR:
        new_value = float(ph.ig_enthalpy_mole) * 1e-3
    elif prop_name == SURFACETENSION_VAR:
        new_value = float(ph.surf_tens)
    elif prop_name == RXNBASEH_VAR:
        new_value = 0 # float(-ph.ig_enthalpy_form_mole*1e-3)
    # elif prop_name == MOLEFLOW_VAR:
    #     new_value = float(ph.flow_sum_mole)
    # elif prop_name == MASSFLOW_VAR:
    #     new_value = float(ph.flow_sum_mole * ph.mw)
    else:
        print('ALERT:', prop_name)
        raise NotImplementedError

    return new_value


def perform_flash(hnd, bulkComp, given_vars, given_vals):
    flash_params = dict(flow_sum_basis=None, flow_sum_value=None,
                        frac_basis=None, frac_value=None,
                        temp=None, press=None,
                        vol_basis=None, vol_value=None,
                        vap_frac_value=None, vap_frac_basis=None,
                        deg_subcool=None, deg_supheat=None,
                        enthalpy_basis=None, enthalpy_value=None,
                        entropy_basis=None, entropy_value=None,
                        int_energy_basis=None, int_energy_value=None, previous=None)

    flash_params['frac_basis'] = 'mole'
    flash_params['frac_value'] = np.array(bulkComp)/np.sum(bulkComp)
    flash_params['flow_sum_basis'] = 'mole'
    flash_params['flow_sum_value'] = 1.0
    for i in range(len(given_vars)):
        if given_vars[i] == T_VAR:
            flash_params['temp'] = float(given_vals[i])
        elif given_vars[i] == P_VAR:
            flash_params['press'] = float(given_vals[i]) * 1e3
        elif given_vars[i] == H_VAR:
            flash_params['enthalpy_value'] = float(given_vals[i]) * 1e3
            flash_params['enthalpy_basis'] = 'mole'
        elif given_vars[i] == S_VAR:
            flash_params['entropy_value'] = float(given_vals[i]) * 1e3
            flash_params['entropy_basis'] = 'mole'
        elif given_vars[i] == VPFRAC_VAR:
            flash_params['vap_frac_value'] = float(given_vals[i])
            flash_params['vap_frac_basis'] = 'mole'

    prov_flash_results = hnd.flash(**flash_params)
    return prov_flash_results


class Sim21StoreInfoThermoCase(object):
    """Groups all the information of a thermodynamic case into one single object that can be sored by pickle"""

    def __init__(self):
        self.version = 1
        self.pkgName = ''
        self.cmps = []
        self.fshSets = None
        self.hypoDescs = None
        self.ipVals = {}  # an ipval can be accessed like self.ipVals[matrName][paneName][i][j]
        self.thCaseObj = None
        self.response = ''
        self.internalData = ''
        self.oilData = ''


class VMGStoreInfoOfProvider(object):
    """Groups all the information of a thermodynamic provider into one single object that can be sored by pickle"""

    def __init__(self):
        self.name = None
        self.parent = None


class ThermoInterface(object):
    """Main class of the interface"""

    def __init__(self):
        """Initializes the package"""
        self.gPkgHandles = {}  # Values are tuples with handle, proppkg creation string and the thermoCase object
        self.propHandler = None
        self.flashSettingsInfoDict = None
        self.flashSettings = {}  # List of values for flash settings
        self.parent = None
        self.name = 'Sim21Thermo'
        self.version = 1
        self._common_property_names = None
        self._common_array_property_names = None

    def Clone(self, thCase, newCaseName):
        if thCase in list(self.gPkgHandles.keys()):
            storeObj = self._CreateAStoreThCaseObj(thCase)
            storeObj.thCaseObj = None
            # This version cannot clone oil data
            # Need to rethink whether oil data are thermo case specific or not
            # If so, need to clone also the Sim42 BasicOilObjects
            storeObj.oilData = ''
            self._CreateAVMGThCase(newCaseName, storeObj)
            return newCaseName

    def __getstate__(self):
        """return info to store"""
        store = {}
        for key in list(self.gPkgHandles.keys()):
            # Wrap all the crucial information into one "pickable" object
            storeObj = self._CreateAStoreThCaseObj(key)
            store[key] = storeObj

        providerStoreObj = VMGStoreInfoOfProvider()
        providerStoreObj.parent = self.parent
        providerStoreObj.name = self.name

        return store, self.propHandler, providerStoreObj

    def __setstate__(self, oldState):
        """build packages from saved info"""
        if len(oldState) == 2:
            (store, self.propHandler) = oldState
            self.name = 'Sim21Thermo'
            self.parent = None
        else:
            (store, self.propHandler, providerStoreObj) = oldState
            self.name = providerStoreObj.name
            self.parent = providerStoreObj.parent

        self.gPkgHandles = {}
        self.flashSettings = {}
        self.flashSettingsInfoDict = None
        for key in list(store.keys()):
            storeObj = store[key]
            self._CreateAVMGThCase(key, storeObj)

        self.version = 1

    def _CreateAStoreThCaseObj(self, thName):
        """Returns an object with all the neccessary information to save a thermodynamic case"""
        storeObj = Sim21StoreInfoThermoCase()
        storeObj.version = self.version
        storeObj.pkgName = self.gPkgHandles[thName][1]
        storeObj.cmps = self.GetSelectedCompoundNames(thName)
        storeObj.fshSets = self.flashSettings[thName]
        storeObj.thCaseObj = self.gPkgHandles[thName][2]

        # load cmps into a var
        cmps = storeObj.cmps
        nuCmps = len(cmps)

        # add Hypo handling
        descs = []
        for idx in range(nuCmps):
            family = self.GetSelectedCompoundProperties(thName, idx, 'ChemicalFamily')[0]

            if cmps[idx][-1] == '*' or family.lower() == 'oil':
                descs.append(self.GetSelectedCompoundProperties(thName, idx, 'CreationInfo')[0])
            else:
                descs.append('')

        storeObj.hypoDescs = descs

        # Now add the IP stuff
        # import time
        # print time.asctime(), time.time()
        ipVals = {}
        for ipMatrName in self.GetIPMatrixNames(thName):
            ipVals[ipMatrName] = {}
            paneNames = self.GetIPPaneNames(thName, ipMatrName)
            for paneIdx in range(len(paneNames)):
                paneName = paneNames[paneIdx]
                try:
                    ipVals[ipMatrName][paneName] = self.GetIPValues(thName, ipMatrName, paneIdx)
                except:
                    ipVals[ipMatrName][paneName] = []
                # lstOfLsts = ipVals[ipMatrName][paneName]
                # Save the minimum amount of values (i.e. do not save ij and ji)
                # for i in range(nuCmps):
                # lstOfLsts.append([])
                # cmpName1 = cmps[i]
                # for j in range(nuCmps):
                # cmpName2 = cmps[j]
                # val = self.GetIPValue(thName, ipMatrName, cmpName1, cmpName2, paneIdx)
                # lstOfLsts[i].append(val)

        storeObj.ipVals = ipVals
        # print time.asctime(), time.time()
        hnd = self.gPkgHandles[thName][0]
        # Get any other internal property package data that need to be stored
        response = self._VMGCommand(hnd, 'GetStoreData', '')
        if response[1] != '' and response[0] == 0:
            storeObj.internalData = response[1]

        try:
            response = vmg.Oil(hnd, 'Store', '')
            if response[1] != '' and response[0] == 0:
                storeObj.oilData = response[1]
        except:
            pass
        # response = self._VMGCommand(hnd, 'Oil', 'SizeOfResults')
        # if response[1] != '' and response[0] == 0:
        # pass

        return storeObj

    def _CreateAVMGThCase(self, thName, storeObj):
        """Any valid  storeObj can build a new thermo case"""

        # Earlier version doesn't store as one object and has no version number stored

        if isinstance(storeObj, tuple):
            # Old style
            # if the store contains 3 tokens, asume version 0
            if len(storeObj) == 3:
                ver = 0
                (pkgName, cmps, fshSets) = storeObj
                descs = []
            else:
                (ver, pkgName, cmps, fshSets, descs) = storeObj

            if ver == 0:
                self.AddPkgFromName(thName, pkgName)
                for cmp in cmps:
                    try:
                        self.AddCompound(thName, cmp)
                    except:
                        # have to have this so IP errors are ignored
                        pass

            elif ver == 1.0:
                self.AddPkgFromName(thName, pkgName)
                for idx in range(len(cmps)):
                    try:
                        if descs[idx] == '':
                            self.AddCompound(thName, cmps[idx])
                        else:
                            descTuple = GetCompoundPropertyLists(cmps[idx], descs[idx])
                            self.AddHypoCompound(thName, cmps[idx], descTuple)
                    except:
                        pass  # have to have this so IP errors are ignored

            for fshSet in list(fshSets.keys()):
                self.SetFlashSetting(thName, fshSet, fshSets[fshSet])

        else:
            ver = storeObj.version

            if ver >= 2.0:
                pkgName = storeObj.pkgName
                cmps = storeObj.cmps
                fshSets = storeObj.fshSets
                descs = storeObj.hypoDescs
                ipVals = storeObj.ipVals

                nuCmps = len(cmps)

                # Prop pkg and compounds
                self.AddPkgFromName(thName, pkgName)
                hnd = self.gPkgHandles[thName][0]
                AddCompound = vmg.AddCompound
                AddHypoCompound = self.AddHypoCompound
                for idx in range(nuCmps):
                    try:
                        if descs[idx] == '':
                            AddCompound(hnd, cmps[idx])
                            # self.AddCompound(thName, cmps[idx])
                        else:
                            descTuple = GetCompoundPropertyLists(cmps[idx], descs[idx])
                            AddHypoCompound(thName, cmps[idx], descTuple)
                    except:
                        pass  # have to have this so IP errors are ignored

                # Flash settings
                for fshSet in list(fshSets.keys()):
                    self.SetFlashSetting(thName, fshSet, fshSets[fshSet])

            hnd = self.gPkgHandles[thName][0]
            # import time
            # print time.asctime(), time.time()
            if 3.0 <= ver < 7.0:
                # IP values

                for ipMatrName, ipPaneDict in list(ipVals.items()):
                    paneNames = self.GetIPPaneNames(thName, ipMatrName)
                    for paneName, lstOfLsts in list(ipPaneDict.items()):
                        if paneName in paneNames:
                            paneIdx = paneNames.index(paneName)
                            lstOfLsts = ipVals[ipMatrName][paneName]

                            vals = array(lstOfLsts, Float)
                            # print 'before'
                            vmg.SetBinaryPairValues(hnd, ipMatrName, paneIdx, vals)
                            # print 'after'
                            # for i in range(nuCmps):
                            # cmpName1 = cmps[i]
                            # for j in range(nuCmps):
                            # cmpName2 = cmps[j]
                            # val = lstOfLsts[i][j]
                            # if i != j:

                            # vmg.SetBinaryPairValue(hnd, ipMatrName, i, j, paneIdx, val)
                    vmg.ResetAijChanged(hnd)

                    # self.SetIPValue(thName, ipMatrName, cmpName1, cmpName2, paneIdx, val)
            elif ver >= 7.0:
                # kij values loading is different as now it is done with map and it also stores i=j values
                for ipMatrName, ipPaneDict in list(ipVals.items()):
                    paneNames = self.GetIPPaneNames(thName, ipMatrName)
                    for paneName, lstOfLsts in list(ipPaneDict.items()):
                        if paneName in paneNames:
                            paneIdx = paneNames.index(paneName)
                            kijArray = ipVals[ipMatrName][paneName]

                            if kijArray:
                                # Load variables that will be used by map call
                                # self._thName = thName
                                # self._nuCmps = nuCmps
                                # self._ipMatrName = ipMatrName
                                # self._paneIdx = paneIdx
                                # self._hnd = self.gPkgHandles[thName][0]
                                # self._cmpIdx1 = None

                                # print 'beforenew'
                                # Load everything with map
                                vmg.SetBinaryPairValues(hnd, ipMatrName, paneIdx, kijArray)

                                # map(self._SetIPValsFromList, kijArray, (range(nuCmps)))
                                # print 'afternew'
                                # Reset this
                    vmg.ResetAijChanged(hnd)

                    # No real need to delete temporary member variables
                    ##del self._thName
                    ##del self._nuCmps
                    ##del self._ipMatrName
                    ##del self._paneIdx
                    ##del self._hnd
                    ##del self._cmpIdx1
            # print time.asctime(), time.time()
            if ver <= 3.0:
                self.propHandler.SetSimCommonArrayPropertyNames([])

            # If it stored a thCaseObj, then use it and overwrite the one that got
            # created automatically in this method
            if hasattr(storeObj, 'thCaseObj'):
                obj = self.gPkgHandles[thName]
                if storeObj.thCaseObj:
                    self.gPkgHandles[thName] = (obj[0], obj[1], storeObj.thCaseObj)

            hnd = self.gPkgHandles[thName][0]
            # restore the property pacakge internal data
            if hasattr(storeObj, 'internalData'):
                dat = storeObj.internalData
                if dat:
                    obj = self.gPkgHandles[thName]
                    self._VMGCommand(hnd, 'RecallData', str(dat))

            try:
                if hasattr(storeObj, 'oilData'):
                    data = storeObj.oilData
                    if not data:
                        data = ''
                    response = vmg.Oil(hnd, 'Recall', data)
            except:
                pass

    def _SetIPValsFromList(self, vals, cmpIdx1):
        """Grabs a list of values and matches it to cmpIdx1 as kij values. Useful for map calls"""
        self._cmpIdx1 = cmpIdx1
        list(map(self._SetIPValFromIndex, vals, list(range(self._nuCmps))))

    def _SetIPValFromIndex(self, val, cmpIdx2):
        """Grabs an ip index and matches it with the active idx and sets the kij value. Useful for map calls"""

        vmg.SetBinaryPairValue(self._hnd, self._ipMatrName, self._cmpIdx1, cmpIdx2, self._paneIdx, val)
        # vmg.ResetAijChanged(self._hnd)

    def CleanUp(self):
        for pkg in list(self.gPkgHandles.values()):
            pkg[2].CleanUp()

        self.gPkgHandles = {}
        self.flashSettingsInfoDict = None
        self.propHandler = None
        self.flashSettingsInfoDict = None
        self.parent = None

    def SetParent(self, parent):
        """Should be a thermo admin instance but is not inforced"""
        if self.parent != parent:
            for thCaseName, obj in list(self.gPkgHandles.items()):
                thCaseObj = ThermoCase(parent, self.name, thCaseName, obj[1])
                self.gPkgHandles[thCaseName] = (obj[0], obj[1], thCaseObj)

        self.parent = parent

    def GetPath(self):
        if self.parent:
            return self.parent.GetPath() + '.' + self.name
        return None

    def GetParent(self):
        return self.parent

    def SetName(self, name):
        self.name = name

    def GetName(self):
        return self.name

    def DeleteObject(self, obj):
        if isinstance(obj, ThermoCase):
            thAdmin = obj.thermoAdmin
            thName = obj.case
            provider = obj.provider

            if thAdmin != self.parent:
                raise SimError('ThAdminMismatch', (str(obj),))

            # First get rid of the thCase from the unit ops
            unitOps = obj.GetUnitOps()
            if thName in list(self.gPkgHandles.keys()) and provider == self.name:
                for uo in unitOps:
                    uo.thCaseObj = None

                # Now delete through the thermo admin!!!!
                self.parent.DeleteThermoCase(provider, thName)

    def GetContents(self):
        results = []
        for thCaseName, myInfo in list(self.gPkgHandles.items()):
            results.append((thCaseName, myInfo[2]))
        return results

    def GetObject(self, desc):
        """Return the thermo case if it exists"""
        if desc in self.gPkgHandles:
            return self.gPkgHandles[desc][2]

        return None

    def MergeProvider(self, fProv):
        """Merge the incoming provider into this thermo provider"""
        self.gPkgHandles.update(fProv.gPkgHandles)
        self.flashSettings.update(fProv.flashSettings)

    def GetAvThCaseNames(self):
        """List of the available thermo cases for a specified provider"""
        return list(self.gPkgHandles.keys())

    def GetAvPropPkgNames(self):
        """List of avilable property packages for a specified provider"""
        pkgs = ['Peng-Robinson',
                'PengRobinson',
                'PR',
                'Soave-Redlich-Kwong',
                'SoaveRedlichKwong',
                'SRK']
        return pkgs

    def AddPkgFromName(self, thName, pkgName):
        """Selects a property packages for a specified thermo case

        thName -- Name of the thermo case
        pkgName -- String with the th pkg name. If one pkg per phase,
                   then separate the th pkg names with a space. Order: Vap, liq

        """
        self.DeleteThermoCase(thName)
        print('pkgName:', pkgName)

        pkgName = pkgName.upper()
        if pkgName in [i.upper() for i in ('Peng-Robinson', 'PengRobinson', 'PR')]:
            hnd = PengRobinson()
        elif pkgName in [i.upper() for i in ('Soave-Redlich-Kwong', 'SoaveRedlichKwong', 'SRK')]:
            hnd = SoaveRedlichKwong()
        else:
            raise NotImplementedError

        thCaseObj = None
        if self.parent:
            thCaseObj = ThermoCase(self.parent, self.name, thName, pkgName)

        self.gPkgHandles[thName] = (hnd, pkgName, thCaseObj)

        # Init flash settings with default values
        d = self.flashSettingsInfoDict
        if d is not None:
            self.flashSettings[thName] = copy.copy(d)

        return thCaseObj

    def ReplacePkgFromName(self, thName, pkgName):
        """Replaces property packages in an existing thermo case

        thName -- Name of the thermo case
        pkgName -- String with the th pkg name. If one pkg per phase,
                   then separate the th pkg names with a space. Order: Vap, liq
        """
        hnd = self.gPkgHandles[thName][0]
        thCase = self.gPkgHandles[thName][2]

        self.gPkgHandles[thName] = (hnd, pkgName, thCase)

        pkgSplit = string.split(pkgName)
        if ' ' in pkgName and (pkgSplit[0] not in pkgSplit[1:]):
            hnd = vmg.ReplaceAggregatePkgFromName(pkgName, hnd)
        else:
            hnd = vmg.ReplacePkgFromName(pkgSplit[0], hnd)
        thCase.package = pkgName
        self.gPkgHandles[thName] = (hnd, pkgName, thCase)

    def ChangeThermoCaseName(self, oldThName, newThName):
        """Change the name of a thermo case"""
        avThCases = self.GetAvThCaseNames()
        if (newThName in avThCases) or (not oldThName in avThCases):
            return
            # Should rise an error

        self.gPkgHandles[newThName] = self.gPkgHandles[oldThName]
        self.flashSettings[newThName] = self.flashSettings[oldThName]
        del self.gPkgHandles[oldThName]
        del self.flashSettings[oldThName]

    def DeleteThermoCase(self, thName):
        """Deletes a thermo case"""
        if thName in self.gPkgHandles:
            del self.gPkgHandles[thName]
            if self.flashSettings is not None and thName in self.flashSettings:
                del self.flashSettings[thName]

    def GetPropPkgString(self, thName):
        """Retrives a string with the selected property package name/s"""
        if thName in self.gPkgHandles:
            return self.gPkgHandles[thName][1]

    def CheckThermoVersion(self):
        # Check the version of the SeaPkg.dll
        pass

    # IP methods
    def GetIPMatrixNames(self, thName):
        """Returns the names of the IP Matrices used by a property package"""
        ipInfo = string.strip(vmg.GetInteractionParameterMatrixNames(self.gPkgHandles[thName][0]))
        ipInfo = string.split(ipInfo, ';')
        matrixNames = []
        for i in ipInfo:
            try:
                matrixNames.append(string.split(i)[1])
            except:
                pass
        return matrixNames

    def GetNuIPPanes(self, thName, ipMatrName):
        """Returns the amount of panes for an IP matrix (aij, bij,...nij)"""
        ipInfo = string.split(vmg.GetInteractionParameterMatrixNames(self.gPkgHandles[thName][0]))
        if ipMatrName in ipInfo:
            ipMatrNameIdx = ipInfo.index(ipMatrName)
            nuIPPanes = int(ipInfo[ipMatrNameIdx + 1])
        else:
            nuIPPanes = None

        return nuIPPanes

    def GetIPPaneNames(self, thName, ipMatrName):
        """Returns the names of the panes for an IP matrix (aij, bij,...nij)"""
        # vmg method Returns a string that looks like this
        # SeaStdPengRobinsonZFactor SeaStdAdvPengRobinson 3 kij0 kij1 kij2 ; SEAPRPenelouxMathias SeaMathiasDensity 2 aij bij ;
        ipInfo = string.split(vmg.GetInteractionParameterMatrixNames(self.gPkgHandles[thName][0]))
        # strip out the file name, incase it is identical to a kij name
        paneNames = []
        if ipMatrName in ipInfo:
            ipMatrNameIdx = ipInfo.index(ipMatrName)
            nuIPPanes = int(ipInfo[ipMatrNameIdx + 1])
            for paneNameIdx in range(nuIPPanes):
                paneNames.append(ipInfo[ipMatrNameIdx + 2 + paneNameIdx])

        return paneNames

    def GetIPValues(self, thName, ipMatrName, pane):
        try:
            hnd = self.gPkgHandles[thName][0]
            vals = vmg.GetBinaryPairValues(hnd, ipMatrName, pane)
            return vals
        except:
            return None

    def GetIPValue(self, thName, ipMatrName, cmpName1, cmpName2, pane):
        """Returns the IP value of a specific pane for two compounds"""
        cmpNames = self.GetSelectedCompoundNames(thName)
        try:
            idx1 = cmpNames.index(cmpName1)
            idx2 = cmpNames.index(cmpName2)
            hnd = self.gPkgHandles[thName][0]
            val = vmg.GetBinaryPairValue(hnd, ipMatrName, idx1, idx2, pane)
            return val
        except:
            return None

    def SetIPValue(self, thName, ipMatrName, cmpName1, cmpName2, pane, value):
        """Sets the IP value of a specific pane for two compounds"""
        cmpNames = self.GetSelectedCompoundNames(thName)
        try:
            idx1 = cmpNames.index(cmpName1)
            idx2 = cmpNames.index(cmpName2)
            hnd = self.gPkgHandles[thName][0]
            vmg.SetBinaryPairValue(hnd, ipMatrName, idx1, idx2, pane, value)
            vmg.ResetAijChanged(hnd)
        except:
            try:
                self.parent.InfoMessage('CantSetIP', (value, cmpName1, cmpName2))
            except:
                pass
            return None

    def GetIPInfo(self, thName, ipMatrName, cmpName1, cmpName2):
        cmpNames = self.GetSelectedCompoundNames(thName)
        # try:
        idx1 = cmpNames.index(cmpName1)
        idx2 = cmpNames.index(cmpName2)
        hnd = self.gPkgHandles[thName][0]
        val = vmg.GetBinaryPairInformation(hnd, ipMatrName, idx1, idx2)
        return val

    # Compound methods
    def GetAvCompoundNames(self):
        """List of available compounds for a specified provider"""
        return chemsep.available()

    def AddCompound(self, thName, cmp):
        """Adds a compound to a  thermo case"""
        hnd = self.gPkgHandles[thName][0]
        hnd.AddCompound(cmp)

    def AddHypoCompound(self, thName, hypoName, hypoDesc):
        """Adds a hypothetical compound to a  thermo case"""
        # translate the property keywords
        hnd = self.gPkgHandles[thName][0]
        print('hypoName:', hypoName)
        print('hypoDesc:', hypoDesc)

        # strDescs = CompoundPropNameFromSimToVmg('String', hypoDesc[0])
        # strVals = hypoDesc[1]
        # lngDescs = CompoundPropNameFromSimToVmg('Long', hypoDesc[2])
        # lngVals = hypoDesc[3]

        tags = hypoDesc[4]
        tags_values = hypoDesc[5]

        assert len(tags) == len(tags_values)

        identifier = hypoName
        tb, sg, mw = None, None, None
        for t, v in zip(tags, tags_values):
            if t == 'MolecularWeight':
                mw = float(v)
            elif t == 'NormalBoilingPoint':
                tb = float(v)
            elif t == 'LiquidDensity@298':
                # TODO Fix conversion to specific gravity
                sg = float(v)/1000

        assert tb is not None and sg is not None
        new_hypo = twu.TwuHypo(identifier, tb, sg, mw=mw)
        hnd.AddCompound(identifier, compound_obj=new_hypo)

    def EditCompound(self, thName, cmpIdx, hypoDesc):
        """Adds a hypothetical compound to a  thermo case"""
        # translate the property keywords
        hnd = self.gPkgHandles[thName][0]
        strDescs = CompoundPropNameFromSimToVmg('String', hypoDesc[0])
        strVals = hypoDesc[1]
        lngDescs = CompoundPropNameFromSimToVmg('Long', hypoDesc[2])
        lngVals = hypoDesc[3]
        dblDescs = CompoundPropNameFromSimToVmg('Double', hypoDesc[4])
        dblVals = hypoDesc[5]
        vmg.EditGeneralCompound(hnd, cmpIdx, strDescs, strVals, lngDescs, lngVals, dblDescs, dblVals)

    def DeleteCompound(self, thName, cmp):
        """Removes a compound from a thermo case"""
        hnd = self.gPkgHandles[thName][0]
        hnd.DeleteCompound(cmp)

    def GetSelectedCompoundNames(self, thName):
        """List of selected compounds for a thermo case"""
        # Returns a list of components currently in the thermo case
        hnd = self.gPkgHandles[thName][0]
        comps = hnd.components
        if comps is None:
            return []
        else:
            return [c.identifier for c in comps]

    def GetHypoCompoundNames(self, thName):
        """List of hypothetical compounds"""
        cmps = self.GetSelectedCompoundNames(thName)
        hypos = []
        nuCmps = len(cmps)

        for idx in range(nuCmps):
            family = self.GetSelectedCompoundProperties(thName, idx, 'ChemicalFamily')[0]
            if cmps[idx][-1] == '*' or family.lower() == 'oil':
                hypos.append(cmps[idx])

        return hypos

    def GetCompoundPropertyNames(self, propGroup):
        propNames = []
        if propGroup & CMP_ID_GRP or propGroup is None:
            propNames.extend(self.propHandler.GetCmpSimIDPropertyNames())
        if propGroup & CMP_NO_EQDEP_GRP or propGroup is None:
            propNames.extend(self.propHandler.GetCmpSimFixedPropertyNames())
        if propGroup & CMP_EQDEP_GRP or propGroup is None:
            propNames.extend(self.propHandler.GetCmpSimEqDepPropertyNames())
        return propNames

    def GetCmpSimIDPropertyNames(self):
        _cmpSimIDProps = ['Id', 'Formula', 'Name', 'CASN', 'ChemicalAbstractsServiceNumber', 'ComponentCASN',
                          'MainChemicalFamily', 'SecondaryChemicalFamily', 'ChemicalFamily']
        return _cmpSimIDProps

    def GetCompoundProperties(self, thName, cmpName, propNames):
        """return property(ies) for component. String properties can only be obtained 1 by one"""
        hnd = self.gPkgHandles[thName][0]
        if isinstance(propNames, str):
            propCount = 1
            sProp = propNames
            if sProp in self.propHandler.GetCmpSimIDPropertyNames() and sProp != 'Id':
                raise NotImplementedError
                # TODO Return the compound property from one of
                #     'Id', 'Formula', 'Name', 'CASN', 'ChemicalAbstractsServiceNumber', 'ComponentCASN',
                #     'MainChemicalFamily', 'SecondaryChemicalFamily', 'ChemicalFamily'
                # try:
                #     return vmg.CompoundStringProperty(hnd, cmpName, sProp)
                # except:
                #     return None
        else:
            propCount = len(propNames)
            sProp = ''
            for prop in propNames:
                sProp += prop + ' '
        try:
            return vmg.CompoundDoubleProperty(hnd, cmpName, sProp, propCount)
        except:
            propsOut = []
            for i in propNames:
                try:
                    propsOut.append(vmg.CompoundDoubleProperty(hnd, cmpName, propNames[i], 1)[0])
                except:
                    propsOut.append(None)
            return propsOut

    def GetSelectedCompoundProperties(self, thName, cmpNo, propNames):
        """return property(ies) for component number cmpNo"""
        hnd = self.gPkgHandles[thName][0]
        if isinstance(propNames, str):
            propNames = [propNames]

        # if type('s') == type(propNames):
        #     propCount = 1
        #     sProp = propNames
        # else:
        #     propCount = len(propNames)
        #     sProp = ''
        #     for prop in propNames:
        #         sProp += prop + ' '
        #
        # sProp = string.strip(sProp)

        # if propCount == 1 and sProp in GetSimHypoStrings():
        #     # get a single string property
        #     strProp = vmg.SelectedCompoundStringProperty(hnd, cmpNo, sProp)
        #     return [strProp]
        # else:
        # get one or more double properties
        values = []
        for prop in propNames:
            given_comp = hnd.components[cmpNo]
            if prop == MOLEWT_VAR:
                values.append(hnd.mw[cmpNo])
            elif prop == IDEALGASENTHALPY_FUNC_VAR:
                values.append(lambda temp: given_comp.ig_enthalpy_mole(temp)*1e-3)
            elif prop == IDEALGASCP_COEFFS_VAR:
                values.append(given_comp.ig_cp_mole_coeffs)
            elif prop == IDEALGAS_ENTHALPY_FORMATION_VAR:
                values.append(given_comp.ig_enthalpy_form_mole)
            elif prop == IDEALGAS_ENTHALPY_SCALING_VAR:
                values.append(1e-3)
            elif prop == IDEALGAS_ENTHALPY_REF_TEMP_VAR:
                values.append(given_comp.ig_temp_ref)
            else:
                raise NotImplementedError

            # return vmg.SelectedCompoundDoubleProperty(hnd, cmpNo, sProp, propCount)

        return values

    def ExchangeCompound(self, thName, cmp1Name, cmp2Name):
        """ exchange the position of cmp1 and cmp2 in the property package"""
        hnd = self.gPkgHandles[thName][0]
        hnd.ExchangeCompound(hnd, cmp1Name, cmp2Name)

    def MoveCompound(self, thName, cmp1Name, cmp2Name):
        """ move cmp1 before cmp2"""
        hnd = self.gPkgHandles[thName][0]
        cmp1Name = cmp1Name.upper()
        cmp2Name = cmp2Name.upper()

        if cmp2Name == '$':
            cmps = self.GetSelectedCompoundNames(thName)
            cmp2Name = cmps[-1]
            cmp2Name = cmp2Name.upper()

        hnd.MoveCompound(cmp1Name, cmp2Name)

    # Stream methods ######################################################################################
    def GetPropertyNames(self):
        """Returns a list of supported properties"""
        _simprops = [T_VAR,
                     P_VAR,
                     MOLEWT_VAR,
                     ZFACTOR_VAR,
                     MOLARV_VAR,
                     MASSDEN_VAR,
                     H_VAR,
                     S_VAR,
                     CP_VAR,
                     CV_VAR,
                     GIBBSFREEENERGY_VAR,
                     HELMHOLTZENERGY_VAR,
                     IDEALGASENTHALPY_VAR,
                     IDEALGASENTROPY_VAR,
                     IDEALGASCP_VAR,
                     RESIDUALENTHALPY_VAR,
                     RESIDUALENTROPY_VAR,
                     RESIDUALCP_VAR,
                     RESIDUALCV_VAR,
                     VISCOSITY_VAR,
                     THERMOCONDUCTIVITY_VAR,
                     SURFACETENSION_VAR,
                     SPEEDOFSOUND_VAR,
                     ISOTHERMALCOMPRESSIBILITY_VAR,
                     DPDVT_VAR,
                     IDEALGASFORMATION_VAR,
                     IDEALGASGIBBS_VAR,
                     MECHANICALZFACTOR_VAR,
                     INTERNALENERGY_VAR,
                     PH_VAR,
                     RXNBASEH_VAR,
                     STDLIQVOL_VAR,
                     STDLIQDEN_VAR,
                     PSEUDOTC_VAR,
                     PSEUDOPC_VAR,
                     PSEUDOVC_VAR,
                     JT_VAR,
                     CPMASS_VAR,
                     CVMASS_VAR,
                     HMASS_VAR,
                     SMASS_VAR]
        return _simprops

    def GetArrayPropertyNames(self):
        """Returns a list of supported array properties"""
        _sim_array_props = ["Composition",
                            "LnFugacityCoefficient",
                            "LnActivityCoefficient",
                            "LnStandardStateFugacity",
                            "LnFugacity",
                            "IdealKValue",
                            "LnSatFugacity",
                            "MassFraction",
                            "IdealVolumeFraction",
                            "IdealGasGibbs",
                            "StdVolFraction",
                            "StdLiqMolVolPerCmp",
                            "EnvelopeCmp",
                            "MolecularWeightArray"]
        return _sim_array_props

    def SetCommonPropertyNames(self, propList):
        """Sets the common property list"""
        # print('propList:', propList)
        self._common_property_names = propList[:]

    def SetCommonArrayPropertyNames(self, propList):
        """Sets the common array property list"""
        self._common_array_property_names = propList[:]

    def GetCommonPropertyNames(self):
        """Sets the common property list"""
        return self._common_property_names

    def GetCommonArrayPropertyNames(self):
        """Sets the common array property list"""
        return self._common_array_property_names

    def GetSpecialProperty(self, thName, inputData, frac, prop, nuPoints=None):
        """
        Return a special property.
        inputData contains any necessary info required to calculate the required prop
        frac is just the composition
        """
        hnd = self.gPkgHandles[thName][0]
        global glbVmgObjects
        feed = glbVmgObjects.get((hnd, 'feed'), None)
        if feed is None:
            feed = vmg.RegisterObject(hnd, 'feed')
            glbVmgObjects[(hnd, 'feed')] = feed

        hnd = self.gPkgHandles[thName][0]
        try:
            vmgProp = self.propHandler.SpecialPropNamesFromSimToVmg([prop])[0]
        except:
            vmgProp = prop

        outputData, status = None, ""

        if vmgProp in ("BOILINGCURVE", "PROPERTYTABLE"):
            # These properties do not use anything extra
            inputData = inputData.upper()
            vmgProp = vmgProp + ' ' + inputData
            vmg.SetObjectDoubleArrayValues(hnd, feed, seaComposition, frac)
            if vmgProp == "BOILINGCURVE":
                outputData, status = vmg.GetSpecialProperty(hnd, feed, vmgProp)
            else:
                outputData, status = vmg.GetPropertyTable(hnd, feed, vmgProp, nuPoints)

        elif vmgProp in ("hydrate3phasepressure", "hydrate3phasetemperature", "HYDRATECURVE"):
            vmgProp = "%s %s %s" % (vmgProp,
                                    inputData,
                                    ' '.join(map(str, frac)))
            # vmg.SetObjectDoubleArrayValues(hnd, feed, seaComposition, frac)
            outputData, status = vmg.GetSpecialProperty(hnd, feed, vmgProp)

        elif vmgProp in ("hydrate3phaseformation",):
            try:
                vmgProp = "%s %s %s" % (vmgProp,
                                        ' '.join(map(str, inputData)),
                                        ' '.join(map(str, frac)))
                outputData, status = vmg.GetSpecialProperty(hnd, feed, vmgProp)
            except:
                status = "Error"

        else:
            try:
                # Generic implementation
                if not isinstance(inputData, str):
                    try:
                        inputData = ' '.join(map(str, inputData))
                    except:
                        inputData = str(inputData)
                vmgProp = vmgProp + ' ' + inputData.upper()
                vmg.SetObjectDoubleArrayValues(hnd, feed, seaComposition, frac)
                outputData, status = vmg.GetSpecialProperty(hnd, feed, vmgProp)
            except:
                pass
        return outputData, status

    def GetProperties(self, thName, inProp1, inProp2, phase, frac, propList):
        """
        Return a list of properties corresponding to the types in propList.
        Two intensive variables must be specified (inProp1 and inProp2). Each of
        these is a tuple with the first member being a string property type.
        The second member can be either a scalar or array variable.
        If the input variables and  phase are scalars and frac a one
        dimensional composition array, then each member of the return list will be a scalar.
        If the input variables, phase are Numeric.arrays and frac a two dimensional
        array with one composition per row, then the return value will be a 2 dim
        Numeric.array
        """
        hnd = self.gPkgHandles[thName][0]

        # prop1Type, prop2Type = self.propHandler.PropNamesFromSimToVmg((inProp1[0], inProp2[0]))
        prop1 = inProp1[1]
        prop2 = inProp2[1]

        if not isinstance(prop1, np.ndarray):
            if len(propList) == 1 and 'MolecularWeight' in propList:
                phase_comp = np.array(frac)
                return [np.dot(hnd.mw, phase_comp/sum(phase_comp))]
            if len(propList) == 1 and RXNBASEH_VAR in propList:
                return 0 # [-np.dot(frac, [c.ig_enthalpy_form_mole for c in hnd.components])*1e-3]

            needsFlash = False
            if phase == OVERALL_PHASE:
                needsFlash = True

            # phase = self.propHandler.PhaseNameFromSimToVmg(phase)

            if (phase == OVERALL_PHASE) or (T_VAR not in (inProp1[0], inProp2[0])) or (
                P_VAR not in (inProp1[0], inProp2[0])):
                needsFlash = True

            if not needsFlash:
                phase_temp, phase_press = None, None
                for i in (inProp1, inProp2):
                    if i[0] == T_VAR:
                        phase_temp = float(i[1])
                    elif i[0] == P_VAR:
                        phase_press = float(i[1]) * 1e3
                    else:
                        raise NotImplementedError

                phase_comp = np.array(frac)
                phase_comp /= sum(phase_comp)
                desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq', SOLID_PHASE: 'solid'}[phase]
                ph = hnd.phase(phase_temp, phase_press, phase_comp, desired_phase)
            else:
                bulkComp = np.array(frac)
                bulkComp /= sum(bulkComp)
                given_vars = (inProp1[0], inProp2[0])
                given_vals = (inProp1[1], inProp2[1])

                prov_flash_results = perform_flash(hnd, bulkComp, given_vars, given_vals)
                if phase == OVERALL_PHASE:
                    ph = prov_flash_results
                else:
                    desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq'}[phase]
                    ph = prov_flash_results[desired_phase]

            values = []
            for prop_name in propList:
                new_value = extract_property_from_phase(prop_name, ph)
                values.append(new_value)

            values = np.array(values)

        else:
            valueArrays = []  # zeros((len(prop1), len(propIDs)), Float)
            for i, prop1_value in enumerate(prop1):
                needsFlash = False
                if isinstance(phase, np.ndarray):
                    chosen_phase = phase[i]
                else:
                    chosen_phase = phase

                if chosen_phase == OVERALL_PHASE:
                    needsFlash = True

                if (chosen_phase == OVERALL_PHASE) or (T_VAR not in (inProp1[0], inProp2[0])) or (
                    P_VAR not in (inProp1[0], inProp2[0])):
                    needsFlash = True

                if needsFlash:
                    given_vars = (inProp1[0], prop1_value)
                    if isinstance(prop2, np.ndarray):
                        prop2_value = prop2[i]
                    else:
                        prop2_value = float(prop2)

                    given_vals = (inProp1[1], prop2_value)
                    if len(np.shape(frac)) == 2:
                        bulkComp = frac[i][:]
                    else:
                        bulkComp = frac[:]

                    bulkComp /= sum(bulkComp)
                    prov_flash_results = perform_flash(hnd, bulkComp, given_vars, given_vals)
                    if isinstance(phase, np.ndarray):
                        chosen_phase = phase[i]
                    else:
                        chosen_phase = phase

                    if chosen_phase == OVERALL_PHASE:
                        ph = prov_flash_results
                    else:
                        desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq'}[chosen_phase]
                        ph = prov_flash_results[desired_phase]

                else:
                    if isinstance(prop2, np.ndarray):
                        prop2_value = prop2[i]
                    else:
                        prop2_value = float(prop2)

                    if isinstance(phase, np.ndarray):
                        chosen_phase = phase[i]
                    else:
                        chosen_phase = phase

                    if len(np.shape(frac)) == 2:
                        phase_comp = frac[i]
                    else:
                        phase_comp = frac

                    phase_comp = np.copy(phase_comp)
                    phase_comp /= sum(phase_comp)

                    phase_temp, phase_press = None, None
                    for j in (inProp1, inProp2):
                        if j[0] == T_VAR:
                            phase_temp = float(prop1_value)
                        elif j[0] == P_VAR:
                            phase_press = float(prop2_value) * 1e3
                        else:
                            raise NotImplementedError

                    desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq', SOLID_PHASE: 'solid'}[chosen_phase]
                    ph = hnd.phase(phase_temp, phase_press, phase_comp, desired_phase)

                ind_values = []
                for prop_name in propList:
                    new_value = extract_property_from_phase(prop_name, ph)
                    ind_values.append(new_value)

                valueArrays.append(ind_values)

            values = np.array(valueArrays)

        return values

    def GetArrayProperty(self, thName, inProp1, inProp2, phase, frac, prop):
        """
        return a Numeric array containing properties of the type 'property'
        Two intensive variables must be specified (inProp1 and inProp2). Each of
        these is a tuple with the first member being a string property type.
        The second member can be either a scalar or array variable.
        If the input variables and phase are scalars, frac should be a single
        composition array and a single array of properties is returned.
        If the input variables and phase are Numeric.arrays, then they must be the same length
        and frac must be a 2d Numeric.array with the same number or rows and each row
        must be a composition.  In this case a 2d Numeric array will be returned with
        one set of results per row
        """
        hnd = self.gPkgHandles[thName][0]

        # prop1Type, prop2Type = self.propHandler.PropNamesFromSimToVmg((inProp1[0], inProp2[0]))

        prop1 = inProp1[1]
        prop2 = inProp2[1]

        assert inProp1[0] in (T_VAR, P_VAR)
        assert inProp2[0] in (T_VAR, P_VAR)

        if not isinstance(prop1, np.ndarray):
            if prop == 'MassFraction':
                frac_mole = frac
                frac_mole /= sum(frac_mole)
                frac_mass = frac_mole*hnd.mw
                frac_mass_sum = np.sum(frac_mass)
                frac_mass = frac_mass/frac_mass_sum
                return frac_mass

            if prop == 'StdVolFraction' or prop == 'IdealVolumeFraction':
                frac_mole = frac
                frac_mole /= sum(frac_mole)
                frac_vol = frac_mole*hnd.std_liq_vol_mole
                frac_vol_sum = np.sum(frac_vol)
                frac_vol /= frac_vol_sum
                return frac_vol

            if prop == IDEALGASGIBBS_VAR:
                if inProp1[0] == T_VAR:
                    temp = prop1
                elif inProp2[0] == T_VAR:
                    temp = prop2
                else:
                    raise NotImplementedError

                if inProp1[0] == P_VAR:
                    press = prop1 * 1e3
                elif inProp2[0] == P_VAR:
                    press = prop2 * 1e3
                else:
                    raise NotImplementedError

                components = hnd.components
                return np.array([c.ig_gibbs_mole(temp, press)*1e-3 for c in components])
            else:
                raise NotImplementedError
        else:
            # prop1 is an array
            # prop2 can be array or scalar
            # phase must be the same length as prop1
            # frac must be array?
            # valueArrays = []
            # if 1:
            value_arrays = []
            for i, prop1_value in enumerate(prop1):
                temp_var, press_var = None, None
                if inProp1[0] == T_VAR:
                    temp_var = float(prop1_value)
                elif inProp1[0] == P_VAR:
                    press_var = float(prop1_value) * 1e3

                if isinstance(prop2, np.ndarray):
                    prop2_value = prop2[i]
                else:
                    prop2_value = inProp2[1]

                if inProp2[0] == T_VAR:
                    temp_var = float(prop2_value)
                elif inProp2[0] == P_VAR:
                    press_var = float(prop2_value) * 1e3

                assert None not in (temp_var, press_var)

                if isinstance(phase, np.ndarray):
                    desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq'}[phase[i]]
                else:
                    desired_phase = {VAPOUR_PHASE: 'vap', LIQUID_PHASE: 'liq'}[phase]

                if len(frac.shape) == 2:
                    phase_comp = frac[i]
                else:
                    phase_comp = frac

                phase_comp = np.copy(phase_comp)
                phase_comp /= sum(phase_comp)

                ph = hnd.phase(temp_var, press_var, phase_comp, desired_phase)
                if prop in (LNFUG_VAR, "LnFugacityCoefficient", "LnFugacityCoeff"):
                    value_arrays.append(ph.log_phi)
                else:
                    raise NotImplementedError

            values = np.array(value_arrays)

        return values

    def GetIdealKValues(self, thName, t, p):
        """
        return array of initial K values based on t and p
        t and p can be scalars in which case an array nComp long is returned
        if t and p are Numeric.arrays they must be the same length and a matrix
        with that many rows of nComp columns will be returned
        """

        hnd = self.gPkgHandles[thName][0]
        if isinstance(t, np.ndarray) and isinstance(p, np.ndarray):
            assert len(t) == len(p)
            k_values_list = []
            for i in range(len(t)):
                temp_act = t[i]
                press_act = p[i] * 1e3  # Pressure is always passed in as kPa
                k_values_list.append(hnd.guess_k_value_vle(temp_act, press_act))

            return np.array(k_values_list)
        else:
            t, p = float(t), float(p)
            temp_act, press_act = t, p * 1e3
            return hnd.guess_k_value_vle(temp_act, press_act)

    def GetMolecularWeightValues(self, thName):
        """shortcut method to get the molecular weight vector of a thermo case"""
        hnd = self.gPkgHandles[thName][0]
        return hnd.mw

    # def _PropertiesForIter(self, prop1, prop2, phase, frac):
    #     feed = self._feed
    #     hnd = self._hnd
    #     vmg.SetMultipleObjectDoubleValues(hnd, feed, (self._prop1Type, self._prop2Type, seaPhaseType),
    #                                       (prop1, prop2, phase))
    #     vmg.SetObjectDoubleArrayValues(hnd, feed, seaComposition, frac)
    #     return vmg.GetMultipleObjectDoubleValues(hnd, feed, self._propIDs)
    #
    # def _ArrPropertiesForIter(self, prop1, prop2, phase, frac):
    #     feed = self._feed
    #     hnd = self._hnd
    #     vmg.SetMultipleObjectDoubleValues(hnd, feed, (self._prop1Type, self._prop2Type, seaPhaseType),
    #                                       (prop1, prop2, phase))
    #     vmg.SetObjectDoubleArrayValues(hnd, feed, seaComposition, frac)
    #     return vmg.GetObjectDoubleArrayValues(hnd, feed, self._propID, 0)

    ####################################################################################################

    ##Flash methods ####################################################################################
    def GetFlashSettingsInfo(self, thName):
        """Returns a dictionary with objects describing the flash settings"""
        return self.flashSettingsInfoDict

    def SetFlashSetting(self, thName, settingName, value):
        hnd = self.gPkgHandles[thName][0]
        vmgSettingName = self.flashSettingsInfoDict[settingName].localName
        myVal = value
        if vmgSettingName == seaVapFracSpec:
            myVal = self.flashSettingsInfoDict[settingName].options.index(value)
        vmg.SetFlashSetting(hnd, vmgSettingName, myVal)

        settingsDict = self.flashSettings[thName]
        settingsDict[settingName] = value

    def GetFlashSetting(self, thName, settingName):
        """Get the value of a flash setting"""
        # Should come directly form vmg
        settingsDict = self.flashSettings[thName]
        return settingsDict[settingName]

    def GetPropNamesCapableOfFlash(self, thName):
        """Returns a tuple with prop names that can be used to calculate a flash"""
        """Simulator  names of the properties that can perform a flash"""
        return T_VAR, P_VAR, H_VAR, S_VAR, VPFRAC_VAR

    def Flash(self, thName, cmps, properties, liqPhCount, propList=None, thThermoAdmin=None, nuSolids=0,
              stdVolRefT=None):
        """
        Performs a Flash calculation

        thName -- Name of the thermo case
        cmps -- Instance of Variables.CompoundList
        properties -- Instance of Variables.MaterialPropertyDict
        liqPhCount -- Number of liquid phases

        returns a FlashResults object
        """

        if not cmps.AreValuesReady():
            return None
        fixed = properties.GetNamesOfKnownFixedVars(CANFLASH_PROP)
        if len(fixed) > 2:
            port = list(properties.values())[0].GetParent()
            parentPath = ''
            if port:
                parentPath = port.GetPath()
            raise SimError('OverspecFlash', (parentPath, len(fixed), ' '.join(fixed)))

        calc = properties.GetNamesOfKnownCalcVars(CANFLASH_PROP)
        returnedType = self.DecideTypeOfFlash(fixed, calc)
        if returnedType is None:
            return None

        flType, given_vars = returnedType
        phasesFracs = []

        given_vals = []
        for i in given_vars:
            given_vals.append(properties[i].GetValue())

        hnd = self.gPkgHandles[thName][0]

        liqPhases = max(liqPhCount, 2)  # Minimum 2
        withSolid = min(nuSolids, 1)    # Maximum 1
        initFromInput = 0
        othersCount = 0

        phasesOut = ['vap', 'liq']
        # phasesOut = ['vap', 'liq_1', 'liq_2']
        if liqPhases == 3:
            # phasesOut.append('liq_3')
            pass

        if withSolid:
            if nuSolids > 1:
                thThermoAdmin.InfoMessage('TooManySolidPhases', (nuSolids, thName))

        bulkComp = cmps.GetValues()
        try:
            bulkComp = np.copy(bulkComp)
            bulkComp /= sum(bulkComp)
            # print('bulkComp:', bulkComp)
            prov_flash_results = perform_flash(hnd, bulkComp, given_vars, given_vals)
        except (FlashConvergenceError, ZeroDivisionError):
            raise
            # pass


        # If propList==None get the common properties
        if not propList:
            propsNamesOut = self.GetSimCommonPropertyNames()
        else:
            propsNamesOut = propList

        # Array properties besides composition
        arrPropNamesOut = self.GetSimCommonArrayPropertyNames()

        bulkProps = []
        for prop_name in propsNamesOut:
            # (T_VAR, P_VAR, H_VAR, S_VAR, VPFRAC_VAR, molarV_VAR, ZFACTOR_VAR, MOLE_WT, STDLIQVOL_VAR)
            new_value = extract_property_from_phase(prop_name, prov_flash_results)
            bulkProps.append(new_value)

        bulkArrProps = []
        # sim42 uses the solid as the last phase

        for prop in arrPropNamesOut:
            # given_vals = vmg.GetObjectDoubleArrayValues(hnd, feed, prop, -1)
            # bulkArrProps.append(given_vals)
            raise NotImplementedError

        phasesComposit = []
        phasesArrProps = []
        phasesProps = []
        # keepValue = -1
        for i in phasesOut:
            # composition
            if i in prov_flash_results:
                composit = prov_flash_results[i].comp_mole
                phasesComposit.append(composit)
                phasesFracs.append(prov_flash_results.frac_mole(i))
            else:
                phasesComposit.append([None]*len(hnd.components))
                phasesFracs.append(0)

            # Array props
            # Cannot get bulk array props
            lstArrProps = []
            for prop in arrPropNamesOut:
                # given_vals = vmg.GetObjectDoubleArrayValues(hnd, i, prop, keepValue)
                # lstArrProps.append(given_vals)
                raise NotImplementedError

            phasesArrProps.append(lstArrProps)

            # Phase props
            props_for_phase = []

            for prop_name in propsNamesOut:
                if i in prov_flash_results:
                    ph = prov_flash_results[i]
                    # (T_VAR, P_VAR, H_VAR, S_VAR, VPFRAC_VAR, molarV_VAR, ZFACTOR_VAR, MOLE_WT, STDLIQVOL_VAR)
                    new_value = extract_property_from_phase(prop_name, ph)
                else:
                    if prop_name == T_VAR:
                        new_value = prov_flash_results.temp
                    elif prop_name == P_VAR:
                        new_value = prov_flash_results.press*1e-3
                    elif prop_name == VPFRAC_VAR and i == 'vap':
                        new_value = 1.0
                    else:
                        new_value = 0.0

                # if prop_name == ZFACTOR_VAR:
                #     print(ZFACTOR_VAR + ':', i, new_value)

                props_for_phase.append(new_value)

            phasesProps.append(props_for_phase)

        # print('phasesOut', phasesOut)
        # print('phaseFracs:', phasesFracs)
        return FlashResults(propsNamesOut, arrPropNamesOut,
                            bulkComp, bulkProps, bulkArrProps,
                            phasesFracs, phasesComposit, phasesProps, phasesArrProps)

    def DecideTypeOfFlash(self, fixed, calc):
        """Decides which type of flash to perform depending on the info av"""
        # This function needs better ideas
        if len(fixed) == 0:
            lookin = calc  # Flash with calc vars
        elif len(fixed) == 1:
            lookin = None  # Flash with one fixed var and one calc var
        elif len(fixed) == 2:
            lookin = fixed  # Flash with  both fixed vars
        else:
            lookin = None  # Too bad :( , the thing has something wrong

        if lookin is not None:
            if P_VAR in lookin and H_VAR in lookin:
                return {P_VAR, H_VAR}, (P_VAR, H_VAR)
            if T_VAR in lookin and H_VAR in lookin:
                return {T_VAR, H_VAR}, (T_VAR, H_VAR)
            if T_VAR in lookin and P_VAR in lookin:
                return {T_VAR, P_VAR}, (T_VAR, P_VAR)
            if P_VAR in lookin and VPFRAC_VAR in lookin:
                return {P_VAR, VPFRAC_VAR}, (P_VAR, VPFRAC_VAR)
            if T_VAR in lookin and VPFRAC_VAR in lookin:
                return {T_VAR, VPFRAC_VAR}, (T_VAR, VPFRAC_VAR)
            if P_VAR in lookin and S_VAR in lookin:
                return {P_VAR, S_VAR}, (P_VAR, S_VAR)
        if T_VAR in fixed:
            if H_VAR in calc:
                if P_VAR in calc:
                    # Better do a PH flash with both calculated rather than using the spec T
                    return {P_VAR, H_VAR}, (P_VAR, H_VAR)

                return {T_VAR, H_VAR}, (T_VAR, H_VAR)
            if P_VAR in calc:
                return {T_VAR, P_VAR}, (T_VAR, P_VAR)
            if VPFRAC_VAR in calc:
                return {T_VAR, VPFRAC_VAR}, (T_VAR, VPFRAC_VAR)

        if P_VAR in fixed:
            if H_VAR in calc:
                return {P_VAR, H_VAR}, (P_VAR, H_VAR)
            if T_VAR in calc:
                return {P_VAR, T_VAR}, (P_VAR, T_VAR)
            if VPFRAC_VAR in calc:
                return {P_VAR, VPFRAC_VAR}, (P_VAR, VPFRAC_VAR)
            if S_VAR in calc:
                return {P_VAR, S_VAR}, (P_VAR, S_VAR)
        if VPFRAC_VAR in fixed:
            if T_VAR in calc:
                return {VPFRAC_VAR, T_VAR}, (VPFRAC_VAR, T_VAR)
            if P_VAR in calc:
                return {VPFRAC_VAR, P_VAR}, (VPFRAC_VAR, P_VAR)
        if S_VAR in fixed:
            if P_VAR in calc:
                return {S_VAR, P_VAR}, (S_VAR, P_VAR)
        if H_VAR in fixed:
            if P_VAR in calc:
                return {H_VAR, P_VAR}, (H_VAR, P_VAR)

        return None

    def PhaseEnvelope(self, thName, cmps, vapFraction, pressure, maxPoints, pList=None):
        hnd = self.gPkgHandles[thName][0]
        global glbVmgObjects
        feedId = glbVmgObjects.get((hnd, 'feed'), None)
        if feedId is None:
            feedId = vmg.RegisterObject(hnd, 'feed')
            glbVmgObjects[(hnd, 'feed')] = feedId

        ##        hnd = self.gPkgHandles[thName][0]
        ##        feedId = vmg.RegisterObject(hnd, 'feed')
        # bulkComp = cmps.GetValues()
        vmg.DefineObject(hnd, feedId, seaLiquidPhase, 298.15, pressure, cmps, seaSeapp)
        pCount = 0
        if pList: pCount = len(pList)
        nc = len(cmps)
        try:
            results = vmg.PhaseEnvelope(hnd, feedId, pCount, pList, vapFraction, nc, maxPoints)
        except:
            # continue the calculation when failed
            pass
        ##        vmg.UnregisterObject(hnd, feedId)

        # return code, message, points, convert the array into an matrix
        retCode = results[0]
        msg = results[1]
        npt = results[2]
        vars = results[3]

        types = []
        pVals = []
        tVals = []
        kVals = []

        ret = []
        i1 = 0
        for i in range(npt):
            i2 = i1 + nc + 3
            types.append(vars[i1])
            pVals.append(vars[i1 + 1])
            tVals.append(vars[i1 + 2])
            kVals.append(vars[i1 + 3:i2])
            i1 = i2
        return EnvelopeResults(retCode, msg, npt, types, pVals, tVals, kVals)

    #
    # Extra methods ######################################################################################
    def HaveSameValues(self, seq1, seq2):
        """Order doesn't matter"""
        for i in seq1:
            if i not in seq2:
                return False
        return True

    def GetValuesInOrder(self, baseNames, destNames, valsIn):
        """Change the order of values

        baseNames -- List with names of the corresponding values in valsIn
        destName -- Order of the values out
        valsIn -- List of values in the order of baseNames

        Returns valsIn in the order of destNames assuming that valsIn is in the
        order of baseNames.
        Useful when the compounds across different things are the same,
        but in different order

        """
        size = len(baseNames)
        if size != len(destNames) or size != len(valsIn):
            return None
        valsOut = []
        for i in range(size):
            for j in range(size):
                if destNames[i] == baseNames[j]:
                    valsOut.append(valsIn[j])

        return valsOut

    # Oil Methods ######################################################################################

    def CustomCommand(self, thCase, cmd):
        # For VMG command the first token is the command key
        # incoming command tokens are separated by dot
        # returned results (retCode, result string)
        (key, remaining) = string.split(cmd, '.', 1)

        hnd = -1
        if thCase:
            hnd = self.gPkgHandles[thCase][0]

        # convert the tokens separator from '.' to to white space
        remaining = self._ConvertString(remaining)

        # Is this a shortcut method for interaction parameter (kij) processing ?
        if key in self.GetCustomIPCommands():
            try:
                return self.ProcessCustomIPCommand(thCase, key, remaining.split())
            except:
                return 1, ''

        return self._VMGCommand(hnd, key, remaining)

    def GetCustomIPCommands(self):
        return ['SetIPValue', 'GetIPValue', 'GetIPInfo',
                'GetIPInfoFromCmpIdx', 'GetAllIPDataForPairCmpIdx']

    def ProcessCustomIPCommand(self, thCase, cmd, tokens):
        """This are shortcut methods to retrive information quick from interaction parameters"""

        ## DO NOT FORGET TO ADD ANY NEW METHOD TO GetCustomIPCommands !!!

        if cmd == 'SetIPValue':

            # f = open('C:\\temp.txt', 'w')
            # f.write(cmd)
            # f.write('\n')
            # f.write(str(tokens) + '\n')

            # Matrix name
            ipMatrName = tokens[0]
            # f.write(ipMatrName + '\n')

            # Pane name
            paneName = str(tokens[1])
            # f.write(paneName + '\n')
            paneNames = self.GetIPPaneNames(thCase, str(ipMatrName))
            # f.write(str(paneNames) + '\n')
            paneIdx = paneNames.index(paneName)
            # f.write(str(paneIdx) + '\n')

            # List compound names
            cmpNames = self.GetSelectedCompoundNames(thCase)

            # Compound 1 name
            cmpName1 = str(tokens[2])
            if not (cmpName1 in cmpNames) and ('_' in cmpName1):
                cmpName1 = re.sub('_', ' ', cmpName1)
            # f.write(cmpName1 + '\n')

            # Compound 2 name
            cmpName2 = str(tokens[3])
            if not (cmpName2 in cmpNames) and ('_' in cmpName2):
                cmpName2 = re.sub('_', ' ', cmpName2)
            # f.write(cmpName2 + '\n')

            # Value being set
            value = float(tokens[4])
            # f.write(str(value) + '\n\n')
            # f.close()

            # Process it
            self.SetIPValue(thCase, ipMatrName, cmpName1, cmpName2, paneIdx, value)

            # All the unit ops should resolve
            thAdmin = self.GetParent()
            if thAdmin is not None:
                thAdmin.ForgetUnitOpsUsingThermo(self.GetName(), thCase)

            return 0, 'OK'

        if cmd == 'GetIPValue':
            # Matrix name
            ipMatrName = tokens[0]

            # Pane name
            paneName = str(tokens[1])
            paneNames = self.GetIPPaneNames(thCase, str(ipMatrName))
            paneIdx = paneNames.index(paneName)

            # List compound names
            cmpNames = self.GetSelectedCompoundNames(thCase)

            # Compound 1 name
            cmpName1 = str(tokens[2])
            if not (cmpName1 in cmpNames) and ('_' in cmpName1):
                cmpName1 = re.sub('_', ' ', cmpName1)

            # Compound 2 name
            cmpName2 = str(tokens[3])
            if not (cmpName2 in cmpNames) and ('_' in cmpName2):
                cmpName2 = re.sub('_', ' ', cmpName2)

            # Process it
            value = self.GetIPValue(thCase, ipMatrName, cmpName1, cmpName2, paneIdx)

            return 0, value

        if cmd == 'GetIPInfo':
            tknIdx = 0

            # Matrix name
            ipMatrName = tokens[tknIdx]
            tknIdx += 1

            # Pane name
            paneName = str(tokens[tknIdx])
            paneNames = self.GetIPPaneNames(thCase, str(ipMatrName))
            if paneName in paneNames:
                # If paneName is not there, then assume that current token is the name of a compound
                paneIdx = paneNames.index(paneName)
                tknIdx += 1

            # List compound names
            cmpNames = self.GetSelectedCompoundNames(thCase)

            # Compound 1 name
            cmpName1 = str(tokens[tknIdx])
            if not (cmpName1 in cmpNames) and ('_' in cmpName1):
                cmpName1 = re.sub('_', ' ', cmpName1)
            tknIdx += 1

            # Compound 2 name
            cmpName2 = str(tokens[tknIdx])
            if not (cmpName2 in cmpNames) and ('_' in cmpName2):
                cmpName2 = re.sub('_', ' ', cmpName2)
            tknIdx += 1

            # Process it
            value = self.GetIPInfo(thCase, ipMatrName, cmpName1, cmpName2)

            return 0, value

        if cmd == 'GetIPInfoFromCmpIdx':
            tknIdx = 0

            # Matrix name
            ipMatrName = str(tokens[tknIdx])
            tknIdx += 1

            # Pane name
            paneName = str(tokens[tknIdx])
            paneNames = self.GetIPPaneNames(thCase, ipMatrName)
            if paneName in paneNames:
                # If paneName is not there, then assume that current token is the name of a compound
                paneIdx = paneNames.index(paneName)
                tknIdx += 1

            # List compound names
            cmpNames = self.GetSelectedCompoundNames(thCase)

            # Compound 1 name
            cmpName1 = cmpNames[int(tokens[tknIdx])]
            tknIdx += 1

            # Compound 2 name
            cmpName2 = cmpNames[int(tokens[tknIdx])]
            tknIdx += 1

            # Process it
            value = self.GetIPInfo(thCase, ipMatrName, cmpName1, cmpName2)

            return 0, value

        if cmd == 'GetAllIPDataForPairCmpIdx':

            # List compound names
            cmpNames = self.GetSelectedCompoundNames(thCase)

            # Compound 1 name
            cmpIdx1 = int(tokens[0])
            cmpName1 = cmpNames[cmpIdx1]

            # Compound 2 name
            cmpIdx2 = int(tokens[1])
            cmpName2 = cmpNames[cmpIdx2]

            retVal = ''
            ipMatrNames = self.GetIPMatrixNames(thCase)
            ipMatrNames.sort()

            # Loop for k i j
            for ipMatrName in ipMatrNames:
                paneIdx = 0
                for paneName in self.GetIPPaneNames(thCase, ipMatrName):
                    value = self.GetIPValue(thCase, ipMatrName, cmpName1, cmpName2, paneIdx)
                    retVal += '%s %s %i %i %f;' % (ipMatrName, paneName, cmpIdx1, cmpIdx2, value)
                    paneIdx += 1

            # Now loop for k j i
            cmpIdx1, cmpIdx2 = cmpIdx2, cmpIdx1
            cmpName1, cmpName2 = cmpName2, cmpName1
            for ipMatrName in ipMatrNames:
                paneIdx = 0
                for paneName in self.GetIPPaneNames(thCase, ipMatrName):
                    value = self.GetIPValue(thCase, ipMatrName, cmpName1, cmpName2, paneIdx)
                    retVal += '%s %s %i %i %f;' % (ipMatrName, paneName, cmpIdx1, cmpIdx2, value)
                    paneIdx += 1

            # remove last character
            if retVal[-1] == ';': retVal = retVal[:-1]

            return 0, retVal

    def InstallOil(self, thCase, assayObj):
        assayName = assayObj.parentObj.name + '.' + assayObj.name
        cmd = 'Oil.InstallOil.' + assayName
        return self.CustomCommand(thCase, cmd)

    def DeletePseudos(self, thCase, assayObj):
        assayName = assayObj.parentObj.name + '.' + assayObj.name
        cmd = 'Oil.DeletePseudos.' + assayName
        return self.CustomCommand(thCase, cmd)

    def PseudoList(self, thCase, assayObj):
        oilName = assayObj.parentObj.name
        assayName = assayObj.name
        hnd = self.gPkgHandles[thCase][0]
        if self._AssayExists(hnd, oilName, assayName) == 1:
            cmd = 'Oil.PseudoList.' + oilName + '.' + assayName
            results = self.CustomCommand(thCase, cmd)
            return string.split(results[1], ';')
        else:
            return None

    def UpdateOil(self, thCase, assayObj):
        assayName = assayObj.parentObj.name + '.' + assayObj.name
        cmd = 'Oil.UpdateOil.' + assayName
        return self.CustomCommand(thCase, cmd)

    def GetOilComposition(self, thCase, assayObj):
        assayName = assayObj.parentObj.name + '.' + assayObj.name
        cmd = 'Oil.GetOilComposition.' + assayName
        result = self.CustomCommand(thCase, cmd)
        if result[0] >= 0 and result[1] != '':
            cmp = string.split(result[1], ';')
        else:
            cmp = None
        return cmp

    def BlendAssay(self, thCase, blend):
        hnd = self.gPkgHandles[thCase][0]
        nAssay = len(blend.assayNames)
        oilName = blend.parentObj.name
        cmd = 'Blend ' + blend.name + ' ' + oilName + ' ' + str(nAssay)
        # count number of assays
        for assay in blend.assays:
            cmd = cmd + ' ' + assay.name

        result = self._VMGCommand(hnd, 'Oil', cmd)
        if result[0] == 0:
            # ok, retrieve the blend info (light ends, curves) data
            self.UpdateAssayPropertiesFromPkg(thCase, blend)
            blend.IsUpToDate(1)
            blend.state = Oils.AssayStateCut
        return result

    def SetAssayParameterValue(self, thCase, paramObj):
        assayObj = paramObj.parentObj
        if isinstance(assayObj, Oils.Assay):
            oilName = assayObj.parentObj.name
            assayName = assayObj.name
            hnd = self.gPkgHandles[thCase][0]
            if self._AssayExists(hnd, oilName, assayName):
                value = paramObj.GetValue()
                if value and str(value) != '':
                    cmd = 'AssayBulkValue ' + oilName + ' ' + assayName + ' ' + paramObj.name + ' ' + str(value)
                    result = self._VMGCommand(hnd, "Oil", cmd)
                    return result

    def CutAssay(self, thCase, assayObj):
        hnd = self.gPkgHandles[thCase][0]

        oilName = assayObj.parentObj.name
        assayName = assayObj.name
        assayPath = oilName + ' ' + assayObj.name

        # If oil do not exist, create it.
        result = self._VMGCommand(hnd, 'Oil', 'GetOilNames')
        names = string.replace(result[1], ' ', '')
        names = string.split(names, ';')
        if oilName in names:
            # Delete the existing assay, if it exists
            self._DeleteAssay(hnd, oilName, assayName)
        else:
            result = self._VMGCommand(hnd, 'Oil', 'AddOil ' + oilName)

        # Create the assay
        result = self._VMGCommand(hnd, 'Oil', 'AddAssay ' + assayPath)

        # Assay now exists but empty
        # First set all parameters
        exptType = ''
        for key in list(assayObj.parameters.keys()):
            value = assayObj.parameters[key].GetValue()
            if value and str(value) != '':
                if key == 'EXPERIMENT':
                    exptType = value
                cmd = "AssayBulkValue " + assayPath + " " + key + " " + str(value)
                result = self._VMGCommand(hnd, "Oil", cmd)
                if result[0] < 0:
                    return result

        # Next, add the required distillation curve, optional MW and density curves
        # if exptType == '', the default experiment type, TBP,  shall be used
        #    return [-1, 'Experiment type not specified']
        if exptType == 'CHROMATOGRAPH':
            if assayObj.chromatograph is None:
                return [-1, 'Missing chromatograph data']
            else:
                result = self._SpecifyOilExperiment(hnd, 'AddChromoPoint ' + assayPath, assayObj.chromatograph)
        elif assayObj.distCurve is None:
            return [-1, 'Missing distillation Curve']
        else:
            result = self._SpecifyOilExperiment(hnd, 'AddDistillationCurve ' + assayPath, assayObj.distCurve)
        if result[0] < 0:
            return result
        if assayObj.MWCurve is not None:
            result = self._SpecifyOilExperiment(hnd, 'AddMolecularWeightCurve ' + assayPath, assayObj.MWCurve)
            if result[0] < 0:
                return result
        if assayObj.denCurve is not None:
            result = self._SpecifyOilExperiment(hnd, 'AddLiquidDensityCurve ' + assayPath, assayObj.denCurve)
            if result[0] < 0:
                return result

        # Add light ends if exists
        lightEnds = assayObj.lightEnds.lightEndDict
        for key in list(lightEnds.keys()):
            value = lightEnds[key].GetValue()
            if value > 0.0:
                cmd = 'AddLightEnds ' + assayPath + ' ' + key + ' ' + str(value)
                result = self._VMGCommand(hnd, "Oil", cmd)
                if result[0] < 0:
                    return result

        # cut the assay
        cmd = 'Cut ' + assayPath
        result = self._VMGCommand(hnd, "Oil", cmd)
        if result[0] != 0:
            raise SimError('ErrorValue', result[1])
        return result

    def DeleteOilObject(self, thCase, obj):
        hnd = self.gPkgHandles[thCase][0]
        if isinstance(obj, Oils.Assay):
            self._DeleteAssay(hnd, obj.parentObj.name, obj.name)
        elif isinstance(obj, Oils.Oil):
            self._DeleteOil(hnd, obj.name)

    def UndateLightEndsFromPkg(self, hnd, assayObj, path):
        cmd = 'GetInputCurve LightEnds ' + path
        response = self._VMGCommand(hnd, 'Oil', cmd)
        if response[0] == 0:
            var = string.split(response[1], ' ')
            for i in range(0, len(var), 2):
                if var and var[i] != '':
                    name = string.strip(var[i])
                    frac = float(var[i + 1])
                    if frac > 0.0:
                        assayObj.lightEnds.AddObject(frac, name)

    def UndateExperimentFromPkg(self, exptType, hnd, assayObj, path):
        cmd = 'GetInputCurve ' + exptType + ' ' + path
        response = self._VMGCommand(hnd, 'Oil', cmd)
        if response[0] == 0:
            # create the experiment
            curve = Oils.OilExperiment(exptType)
            # give the experiment to the assay
            assayObj.AddObject(curve, exptType)
            # assign the value
            data = string.split(response[1], ' ')
            x = []
            y = []
            for i in range(0, len(data), 2):
                if data[i] and data[i] != '':
                    x.append(float(data[i]))
                    y.append(float(data[i + 1]))
            curve.GetObject('Series0').SetValues(x)
            curve.GetObject('Series1').SetValues(y)

    def UpdateAssayPropertiesFromPkg(self, thCase, assayObj):
        hnd = -1
        if thCase:
            hnd = self.gPkgHandles[thCase][0]
        oilName = assayObj.parentObj.name
        assayName = assayObj.name
        path = oilName + ' ' + assayName
        # get all the assay parameters
        cmd = 'GetAssayBulkProperty ' + path
        paramStr = self._VMGCommand(hnd, 'Oil', cmd)
        params = string.split(paramStr[1], ';')
        for par in params:
            # create the oil parameters
            var = string.split(par, ':')
            addData = 1
            try:
                val = float(var[1])
                if val == VMGUnknown:
                    addData = 0
            except ValueError:
                val = str(var[1])  # leave it as string
            if addData == 1:
                parName = string.strip(var[0])
                paramObj = Oils.OilParameter(GENERIC_VAR, val)
                paramObj.parentObj = assayObj
                paramObj.name = parName
                assayObj.parameters[parName] = paramObj
        # get the light ends
        self.UndateLightEndsFromPkg(hnd, assayObj, path)
        # get the density curve
        self.UndateExperimentFromPkg('DensityCurve', hnd, assayObj, path)
        # get the MW curve
        self.UndateExperimentFromPkg('MWCurve', hnd, assayObj, path)
        # get the distillation curve
        self.UndateExperimentFromPkg('DistillationCurve', hnd, assayObj, path)
        return 1

    def _AssayExists(self, hnd, oilName, assayName):
        # Check if the oil exists
        result = self._VMGCommand(hnd, 'Oil', 'GetOilNames')
        names = string.replace(result[1], ' ', '')
        names = string.split(names, ';')
        if oilName in names:
            # Check if the assay exists
            result = self._VMGCommand(hnd, 'Oil', 'GetAssayNames ' + oilName)
            names = string.replace(result[1], ' ', '')
            names = string.split(names, ';')
            if assayName in names:
                return 1
        return 0

    def _DeleteOil(self, hnd, oilName):
        # Delete the oil if it exists
        result = self._VMGCommand(hnd, 'Oil', 'GetOilNames')
        names = string.replace(result[1], ' ', '')
        names = string.split(names, ';')
        if oilName in names:
            self._VMGCommand(hnd, 'Oil', 'DeleteOil ' + oilName)

    def _DeleteAssay(self, hnd, oilName, assayName):
        # Delete the assay if it exists
        if self._AssayExists(hnd, oilName, assayName):
            result = self._VMGCommand(hnd, 'Oil', 'DeleteAssay ' + oilName + ' ' + assayName)

    def _VMGCommand(self, hnd, key, cmd):
        # command tokens are separated by white spaces
        result = vmg.CustomCommand(hnd, key, cmd)
        if result[0] != 0 and key != 'GetStoreData':
            self.GetParent().InfoMessage('ErrorValue', result[1])
        return result

    def _SpecifyOilExperiment(self, hnd, key, tbl):
        x = tbl.GetSeries(0)
        y = tbl.GetSeries(1)
        result = (0, 'OK')
        if x and y:
            nx = x.GetLen()
            ny = y.GetLen()
            if nx == ny:
                for i in range(nx):
                    if y.GetDataPoint(i) is not None and x.GetDataPoint(i) is not None:
                        cmd = key + ' ' + str(x.GetDataPoint(i)) + ' ' + str(y.GetDataPoint(i))
                        result = self._VMGCommand(hnd, "Oil", cmd)
                        if result[0] < 0:
                            return result
        return result

    def GetSimCommonPropertyNames(self):
        return self.GetCommonPropertyNames()

    def GetSimCommonArrayPropertyNames(self):
        return self.GetCommonArrayPropertyNames()

