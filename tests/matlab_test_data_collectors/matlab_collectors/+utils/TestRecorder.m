classdef TestRecorder < handle
   properties
      record_filename = ""
      records = containers.Map
      step = 0
   end
   methods
       
       function obj = TestRecorder(record_filename)
           obj.record_filename = record_filename;
       end
       
       function obj = recordObject(obj, obj_name, obj_to_record)
           warning('off', 'MATLAB:structOnObject')
           key = ['step_' num2str(obj.step) '___' obj_name];
           obj.records(key) = struct(obj_to_record);
       end

       function obj = recordVariable(obj, var_name, var_value)
           key = ['step_' num2str(obj.step) '___' var_name];
           obj.records(key) = var_value;
       end
       
       function obj = increment(obj)
           obj.step = obj.step + 1;
       end
       
      function obj = saveRecordsToDisk(obj)
          k = keys(obj.records) ;
          val = values(obj.records) ;
          for i = 1:length(obj.records)
              curr_key = k{i};
              curr_val = val{i};
              eval([curr_key, '=curr_val;']);
          end
          total_steps = obj.step;
          save(obj.record_filename, k{:}, 'total_steps');
      end
      
   end
end
