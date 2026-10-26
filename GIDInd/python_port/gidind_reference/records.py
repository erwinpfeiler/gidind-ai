"""Compact, independently recoverable full-solution journal records."""
import base64
import hashlib
import json
import zlib


def encode_record(event, solution):
    raw=json.dumps(solution,allow_nan=False,separators=(',',':')).encode()
    return json.dumps(dict(sequence=event['sequence'],solution_slot=event['solution_slot'],
        sha256=hashlib.sha256(raw).hexdigest(),
        zlib_base64=base64.b64encode(zlib.compress(raw,1)).decode()),separators=(',',':'))+'\n'


def read_records(path):
    """Recover complete records; only an incomplete final line is ignored."""
    active={};count=0;truncated=False
    with open(path,'rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                truncated=True
                break
            record=json.loads(line)
            if record['sequence']!=count:raise ValueError('Invalid solution-record sequence')
            raw=zlib.decompress(base64.b64decode(record['zlib_base64'],validate=True))
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:
                raise ValueError('Solution-record checksum mismatch')
            active[record['solution_slot']]=json.loads(raw)
            count+=1
    return active,dict(complete_records=count,truncated_final_record=truncated)
