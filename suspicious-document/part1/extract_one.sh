#!/bin/bash

file="suspicious-document00027.xml"
suspicious_id=$(basename "$file" .xml)

# Extract plagiarism features using grep and sed
grep 'feature name="plagiarism"' "$file" | while read line; do
    # Extract attributes using sed
    type=$(echo "$line" | sed -n 's/.*type="\([^"]*\)".*/\1/p')
    obfuscation=$(echo "$line" | sed -n 's/.*obfuscation="\([^"]*\)".*/\1/p')
    this_offset=$(echo "$line" | sed -n 's/.*this_offset="\([^"]*\)".*/\1/p')
    this_length=$(echo "$line" | sed -n 's/.*this_length="\([^"]*\)".*/\1/p')
    source_ref=$(echo "$line" | sed -n 's/.*source_reference="\([^"]*\)".*/\1/p')
    source_offset=$(echo "$line" | sed -n 's/.*source_offset="\([^"]*\)".*/\1/p')
    source_length=$(echo "$line" | sed -n 's/.*source_length="\([^"]*\)".*/\1/p')
    
    # Print CSV row
    echo "$source_ref,$suspicious_id,$type,$obfuscation,$source_offset,$source_length,$this_offset,$this_length"
done
