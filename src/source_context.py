######################################################################## Source Context w/ Response Regex filtering 

import re
    
def source_context(response):
    formatted_output = ""
    if hasattr(response, 'metadata'):
        document_info = str(response.metadata)
        combined_info = []

        if "pandas_instruction_str" in response.metadata:
            formatted_output = ""
            return formatted_output
        
        if len(re.findall(r"'page_label': '[^']*'", document_info)) < 1:
            find = re.findall(r"'file_name': '[^']*'", document_info)

            formatted_output = "\n" + ("=" * 60) + "\n" 
            formatted_output += "Context Information\n\n" 
            formatted_output += ",\n".join(map(str, find))
            formatted_output += "\n" + ("=" * 60) + "\n"
            return formatted_output

        else:
            for file_name, page_label in zip(re.findall(r"'file_name': '[^']*'", document_info), re.findall(r"'page_label': '[^']*'", document_info)):
                combined_info.append(f"{file_name}, page : {page_label.split(':')[1]}")
            
            formatted_output = '\n' + '=' * 30 + '\n' + 'Source Context\n\n' + ",\n".join(combined_info) + '\n' + '=' * 30 + '\n'
            return formatted_output
