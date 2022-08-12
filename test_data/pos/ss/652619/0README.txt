#-----------------------------------------------------------------
 Explanation of files in seqid.property.zip,
 where seqid is a number assigned by RaptorX to the query sequence
#-----------------------------------------------------------------

# ------- README files ------------------ #
0README.txt         : README file for Mac OX and Linux system.
0README.rtf         : README file for Windows system.

# ------- main result files ------------- #
seqid.fasta.txt     : The raw sequence from user's input in FASTA format with full description.
seqid.seq.txt       : The user input sequence in FASTA format with invalid amino acid shown as 'X'.
seqid.all.txt       : The master output file that combines all single prediction results.

# ------- single prediction results ----- #
seqid.ss3.txt       : 3-class secondary structure prediction.
seqid.ss3_simp.txt  : A simple summary of 3-class secondary structure prediction in FASTA format.
seqid.ss8.txt       : 8-class secondary structure prediction.
seqid.ss8_simp.txt  : A simple summary of 8-class secondary structure prediction in FASTA format.
seqid.acc.txt       : Solvent accessibility (ACC) prediction.
seqid.acc_simp.txt  : A simple summary of solvent accessibility (ACC) prediction in FASTA format.
seqid.diso.txt      : Disorder prediction.
seqid.diso_simp.txt : A simple summary of disorder prediction in FASTA format.
#==================================================================

#-> note 1
For Mac OX and Linux system, all result files can be opened directly.

For Windows system, all corresponding result files could be found in Windows/ directory with suffix ".rtf",
	and please use WordPad to open these ".rtf" files.


#-> note 2
The master output file has two major parts: the brief prediction summary, and the detail prediction results.

For the first part:
	the first row shows the original description of the user's input sequence;
	the second row shows the user input sequence with invalid amino acid shown as 'X';
	the following 4 rows shows the simple summary of 
		(a) 3-class secondary structure (SS3) prediction, 
		(b) 8-class secondary structure (SS8 )prediction, 
		(c) 3-state solvent accessibility (ACC) prediction, and 
		(d) disorder (DISO )prediction, respectively.

The second part shows the detail prediction results of
	(a) SS3 prediction,
	(b) SS8 prediction,
	(c) ACC prediction, and
	(d) DISO prediction, respectively.


A schematic example is shown below:

>original description
ASDFASDGFAGASG    #-> user input sequence with invalid amino acid shown as 'X'.
HHHHEEECCCCCHH    #-> 3-class secondary structure (SS3) prediction.
HHGGEEELLSSTHH    #-> 8-class secondary structure (SS8 )prediction.
EEMMEEBBEEEBBM    #-> 3-state solvent accessibility (ACC) prediction.
*****......***    #-> disorder (DISO )prediction, with disorder residue shown as '*'.

---------------- details of SS3 prediction ---------------------------

---------------- details of SS8 prediction ---------------------------

---------------- details of ACC prediction ---------------------------

---------------- details of DISO prediction --------------------------


