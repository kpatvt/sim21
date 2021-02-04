#Test thermo cases


#Create any dummy unit ops

s1_Lev1 = Flowsheet.SubFlowsheet()
s2_Lev1 = Flowsheet.SubFlowsheet()


#Now create a first thermo case
$thermo = Sim21Thermo.SRK
/ -> $thermo
thermo + METHANE ETHANE PROPANE

#See who got it
/thermo
s1_Lev1.thermo
s2_Lev1.thermo


#Create one more unit op at the same level and see its thermo
s3_Lev1 = Flowsheet.SubFlowsheet()
s3_Lev1.thermo


#Create a thermo case in the root with the same name
$thermo = Sim21Thermo.PengRobinson
/ -> $thermo
thermo + WATER

#See who got it
/thermo
s1_Lev1.thermo
s2_Lev1.thermo
s3_Lev1.thermo


#Now create separate thermo cases named thermo in one of the childs
s1_Lev1.thermo2 = Sim21Thermo.PR
s1_Lev1.thermo2 + METHANE n-HEXANE
s2_Lev1.thermo = Sim21Thermo.IdealVapLiq
s2_Lev1.thermo + n-BUTANE

#See...
/thermo
s1_Lev1.thermo2
s2_Lev1.thermo
s3_Lev1.thermo


#Now create a child unit ops
/s1_Lev1.s1_Lev2 = Flowsheet.SubFlowsheet()
/s1_Lev1.s2_Lev2 = Flowsheet.SubFlowsheet()
/s1_Lev1.s3_Lev2 = Flowsheet.SubFlowsheet()

/s2_Lev1.s1_Lev2 = Flowsheet.SubFlowsheet()
/s2_Lev1.s2_Lev2 = Flowsheet.SubFlowsheet()
/s2_Lev1.s3_Lev2 = Flowsheet.SubFlowsheet()


/s3_Lev1.s1_Lev2 = Flowsheet.SubFlowsheet()
/s3_Lev1.s2_Lev2 = Flowsheet.SubFlowsheet()
/s3_Lev1.s3_Lev2 = Flowsheet.SubFlowsheet()



#See their thermo
/s1_Lev1.s1_Lev2.thermo2
/s1_Lev1.s2_Lev2.thermo2


/s2_Lev1.s1_Lev2.thermo
/s2_Lev1.s2_Lev2.thermo


/s3_Lev1.s1_Lev2.thermo
/s3_Lev1.s2_Lev2.thermo


#This deletes the spec thermo in child AND parents. Mhhh kind of not consistent
/s2_Lev1.s1_Lev2.thermo =
/thermo
s1_Lev1.thermo2
s2_Lev1.thermo
s3_Lev1.thermo

/s1_Lev1.s1_Lev2.thermo2
/s1_Lev1.s2_Lev2.thermo2


/s2_Lev1.s1_Lev2.thermo
/s2_Lev1.s2_Lev2.thermo


/s3_Lev1.s1_Lev2.thermo
/s3_Lev1.s2_Lev2.thermo


/s3_Lev1.thermo = Sim21Thermo.SRK
/s3_Lev1.thermo + HYDROGEN


/thermo = Sim21Thermo.IdealVapLiqWilson
/thermo
s1_Lev1.thermo2
s2_Lev1.thermo
s3_Lev1.thermo




