import { useQuery } from '@tanstack/react-query';
import { mockTarsApi } from '../services/tarsApi';

export const useProjects = () => {
  return useQuery({
    queryKey: ['projects'],
    queryFn: mockTarsApi.getProjects,
    refetchInterval: 10000, // Refetch every 10 seconds
  });
};
