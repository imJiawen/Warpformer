# This file makes tables for the concepts in this subfolder.
# Be sure to run postgres-functions.sql first, as the concepts rely on those function definitions.
# Note that this may take a large amount of time and hard drive space.

# ASSIGN THE PATH TO mimic-iii HERE
export MIMIC_CODE_DIR='path/to/mimic-code/mimic-iii'

# string replacements are necessary for some queries
export REGEX_DATETIME_DIFF="s/DATETIME_DIFF\((.+?),\s?(.+?),\s?(DAY|MINUTE|SECOND|HOUR|YEAR)\)/DATETIME_DIFF(\1, \2, '\3')/g"
export REGEX_SCHEMA='s/`physionet-data.(mimiciii_clinical|mimiciii_derived|mimiciii_notes).(.+?)`/\2/g'
export CONNSTR='-d mimic'

# this is set as the search_path variable for psql
# a search path of "public,mimiciii" will search both public and mimiciii
# schemas for data, but will create tables on the public schema
export PSQL_PREAMBLE='SET search_path TO public,mimiciii'

echo ''
echo '==='
echo 'Beginning to create tables for MIMIC database.'
echo 'Any notices of the form "NOTICE: TABLE "XXXXXX" does not exist" can be ignored.'
echo 'The scripts drop views before creating them, and these notices indicate nothing existed prior to creating the view.'
echo '==='
echo ''

echo 'Directory 5 of 9: fluid_balance'
{ echo "${PSQL_PREAMBLE}; DROP TABLE IF EXISTS colloid_bolus; CREATE TABLE colloid_bolus AS "; cat $MIMIC_CODE_DIR/concepts/fluid_balance/colloid_bolus.sql; } | sed -r -e "${REGEX_DATETIME_DIFF}" | sed -r -e "${REGEX_SCHEMA}" | psql ${CONNSTR}
{ echo "${PSQL_PREAMBLE}; DROP TABLE IF EXISTS crystalloid_bolus; CREATE TABLE crystalloid_bolus AS "; cat $MIMIC_CODE_DIR/concepts/fluid_balance/crystalloid_bolus.sql; } | sed -r -e "${REGEX_DATETIME_DIFF}" | sed -r -e "${REGEX_SCHEMA}" | psql ${CONNSTR}


echo 'Finished creating tables.'
