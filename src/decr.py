def decrypt_key(encrypted_data, key):
  delimiter = b"~~"  # Choose a delimiter that won't appear in the data or key
  combined_data = encrypted_data + delimiter + key
  return combined_data.decode("latin-1")

encrypted_data =  b'\x18\nO+\x16R0A -$\x11@\nQ%\x0fW\x11\x07\x18Q3\'X#\x0e\x0b\x19"+\x11\x0f(\x14(\n- \x16\x04\x1b\x069G-\x12\x05Z\x14\x16'
key = b'kabirdas'
openai_api_key = decrypt_key(encrypted_data, key)

print(openai_api_key)